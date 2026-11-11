"""Review bounded channel-pilot outputs, including original multi-objective reward."""
from pathlib import Path
from datetime import datetime,timezone
import json,math,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from config.reward import PRIORITY_PROFILES

COLS=["round","task_id","device_id","device_type","class_name","status","selected_tier","execution_tier","reward","success","deadline_ms","deadline_missed","observed_complete_latency_ms","tx_delay_wireless_ms","tx_delay_wired_ms","propagation_total_ms","queue_delay_ms","processing_delay_ms","handover_delay_component_ms","device_battery_energy_j","total_energy_j","battery_capacity_j","device_battery_before_j","device_battery_after_j","current_util","selected_projected_utilization_percent","reward_compute_capacity_norm","reward_capacity_adequacy","reward_device_battery_score","reward_server_battery_score","reward_cost_penalty","reward_mobility_violation","reward_shared_capacity_externality","reward_overutilization_penalty","reward_invalid_action_penalty","throughput_mbps","coverage_outage_flag","network_status","network_failure_reason","data_size_kb","epsilon","remaining_dwell_time_ms"]
TEXT={"task_id","device_type","class_name","status","selected_tier","execution_tier","network_status","network_failure_reason"}

def prepare(d):
    for c in d:
        if c not in TEXT:d[c]=pd.to_numeric(d[c],errors="coerce")
    d["complete"]=d.status.eq("completed")
    d["on_time"]=d.success.eq(1)
    d["selected_entropy"]=0.
    for _,g in d.groupby("class_name",sort=False):
        p=pd.get_dummies(g.selected_tier.fillna("NONE"),dtype=float).rolling(50,min_periods=1).mean()
        d.loc[g.index,"selected_entropy"]=-(p*np.log(p.where(p.gt(0),1))).sum(axis=1)
    return d

def summarize(g):
    ok=g[g.complete]
    c=g.groupby("device_id").battery_capacity_j.first(); eligible=c.gt(0)
    draw=(g.device_battery_before_j-g.device_battery_after_j).clip(lower=0).groupby(g.device_id).sum().reindex(c.index)
    last=g.groupby("device_id").device_battery_after_j.last()
    tiers=ok.execution_tier.value_counts()
    return dict(attempts=len(g),completed=int(g.complete.sum()),completion_pct=100*g.complete.mean(),deadline_success_pct=100*g.on_time.mean(),coverage_outage_observation_pct=100*g.coverage_outage_flag.eq(1).mean(),mean_delay_ms=ok.observed_complete_latency_ms.mean(),p95_delay_ms=ok.observed_complete_latency_ms.quantile(.95),mean_queue_ms=ok.queue_delay_ms.mean(),mean_processing_ms=ok.processing_delay_ms.mean(),mean_wireless_tx_ms=ok.tx_delay_wireless_ms.mean(),mean_wired_tx_ms=ok.tx_delay_wired_ms.mean(),mean_propagation_ms=ok.propagation_total_ms.mean(),mean_handover_ms=ok.handover_delay_component_ms.mean(),device_energy_mj=1000*ok.device_battery_energy_j.mean(),total_energy_j=ok.total_energy_j.mean(),battery_task_depletion_pct=(100*draw[eligible]/c[eligible]).mean(),battery_remaining_pct=(100*last[eligible]/c[eligible]).mean(),headroom_pct=(100-ok.current_util.clip(0,100)).mean(),compute_score=ok.reward_compute_capacity_norm.mean(),success_adjusted_compute_score=g.reward_compute_capacity_norm.where(g.on_time,0).fillna(0).mean(),task_goodput_mbps=ok.throughput_mbps.mean(),mean_reward=g.reward.mean(),selected_entropy_nats=g.selected_entropy.mean(),mean_payload_kb=g.data_size_kb.mean(),dominant_tier=tiers.index[0] if len(tiers) else None)

def main():
    root=Path((ROOT/"output/latest_aiot_channel_pilot.txt").read_text().strip())
    manifest=json.loads((root/"manifest.json").read_text())
    jobs=json.loads((root/"completed_jobs.json").read_text())
    assert len(jobs)==8
    frames=[]; metrics=[]; trends=[]; checks=[]; failures=[]; lossrows=[]
    for job in jobs:
        for phase,details in job["phases"].items():
            p=Path(details["source_csv"])
            d=prepare(pd.read_csv(p,sep=";",usecols=lambda c:c in COLS,low_memory=False))
            assert len(d)==details["decisions"] and d.task_id.nunique()==len(d)
            assert np.isfinite(d.reward).all()
            recon=[]
            for cls,g in d.groupby("class_name"):
                profile=next(x for x in PRIORITY_PROFILES.values() if x["name"]==cls)
                w=profile["objective_weights"]
                delay=(2/(1+g.observed_complete_latency_ms/profile["delay_target_ms"])-1).clip(-1,1)
                mobility=(g.execution_tier.isin(["MEC","MEC_COLLAB","UAV_BS"]) & g.remaining_dwell_time_ms.gt(0) & g.observed_complete_latency_ms.gt(g.remaining_dwell_time_ms)).astype(float)
                mobility_disagreements=int(mobility.ne(g.reward_mobility_violation).sum())
                objective=(w["delay"]*delay+w["device_battery"]*g.reward_device_battery_score+w["server_battery"]*g.reward_server_battery_score+w["compute"]*g.reward_compute_capacity_norm-w["cost"]*g.reward_cost_penalty-w["mobility"]*mobility-w["shared_capacity_externality"]*g.reward_shared_capacity_externality)
                expected=(10*(objective-.2*g.reward_overutilization_penalty).clip(-1,1)-g.reward_invalid_action_penalty).clip(-10,10)
                expected.loc[~g.on_time]=-10
                error=(g.reward-expected).abs().max()
                assert error<1e-5,(p,cls,error)
                meta=dict(band=job["band"],model=job["model"],seed=job["repetition"],phase=phase,task_class=cls)
                metrics.append(dict(**meta,**summarize(g)))
                checks.append(dict(**meta,rows=len(g),max_reward_reconstruction_error=error,logged_mobility_flag_disagreements=mobility_disagreements))
                for reason,z in g[~g.on_time].groupby("network_failure_reason",dropna=False):
                    failures.append(dict(**meta,reason=str(reason),count=len(z)))
                if phase=="train":
                    for label,z in [("first15",g[g["round"].le(15)]),("last15",g[g["round"].gt(45)])]:
                        trends.append(dict(**meta,window=label,mean_reward=z.reward.mean(),completion_pct=100*z.complete.mean(),selected_entropy=z.selected_entropy.mean()))
            d["band"]=job["band"];d["model"]=job["model"];d["seed"]=job["repetition"];d["phase"]=phase
            frames.append(d)
            history=p.parent/"training_metrics.csv"
            if not history.exists():
                history=p.parent/(job["model"]+"_loss_log.csv")
            if history.exists():
                h=pd.read_csv(history)
                if "loss" in h:
                    loss=pd.to_numeric(h.loss,errors="coerce").dropna()
                    if len(loss):
                        assert np.isfinite(loss).all()
                        lossrows.append(dict(band=job["band"],model=job["model"],seed=job["repetition"],phase=phase,loss_count=len(loss),first50_mean=loss.head(50).mean(),last50_mean=loss.tail(50).mean()))
    assert sum(x["rows"] for x in checks)==23040
    # Match initial cohorts and topology within each model and seed.
    for index in (1,2):
        for model in ("random","dqn"):
            a=next(x for x in jobs if x["band"]=="FR1" and x["repetition"]==index and x["model"]==model)
            b=next(x for x in jobs if x["band"]=="FR3" and x["repetition"]==index and x["model"]==model)
            for phase in ("train","evaluation"):
                assert a["phases"][phase]["layout_sha256"]==b["phases"][phase]["layout_sha256"]
                assert a["phases"][phase]["device_mix"]==b["phases"][phase]["device_mix"]
    pooled=[]
    all_data=pd.concat(frames,ignore_index=True)
    for (band,model,phase,cls),g in all_data.groupby(["band","model","phase","class_name"]):
        # Battery depletion uses a unique per-seed device key when pooling.
        g=g.copy();g["device_id"]=g.seed.astype(str)+":"+g.device_id.astype(str)
        pooled.append(dict(band=band,model=model,phase=phase,task_class=cls,seeds=g.seed.nunique(),**summarize(g)))
    out=root/"analysis";out.mkdir(exist_ok=True)
    for name,data in [("class_metrics_by_seed",metrics),("pooled_class_metrics",pooled),("training_windows",trends),("reward_checks",checks),("failure_reasons",failures),("loss_metrics",lossrows)]:pd.DataFrame(data).to_csv(out/(name+".csv"),index=False)
    drift=[p for p,h in manifest["source_hashes"].items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not drift,drift
    validation={"captured_utc":datetime.now(timezone.utc).isoformat(),"completed_jobs":len(jobs),"phases":len(frames),"total_decisions":len(all_data),"original_reward_checks_passed":True,"logged_mobility_flag_disagreements":sum(x["logged_mobility_flag_disagreements"] for x in checks),"matched_layout_and_initial_cohorts":True,"simulator_source_unchanged":True,"caveats":["Two seeds, short training, fixed 32-device workload; not a convergence or full-scale validation.","Original CSV mobility-violation flag disagrees with reward evaluator for some zero-dwell rows; reward reconstruction uses the actual evaluator condition and records the discrepancy count.","Conditional delay/energy/capacity metrics exclude unsuccessful tasks; inspect deadline success alongside them.","Channel profiles differ in bandwidth, fading, gain, blockage, spatial streams and frequency.","Common 6G application deadlines used for both bands; UAV access, lowband anchor, terrestrial node count and geometry fixed.","Physical feasibility masks and class-neutral compute spillover are enabled, as in version 1.1.","Compute is lightly loaded in this small pilot; capacity scores do not prove saturation performance."]}
    (out/"validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
    eval_data=pd.DataFrame(pooled);eval_data=eval_data[eval_data.phase.eq("evaluation")]
    print(eval_data[["band","model","task_class","attempts","completion_pct","deadline_success_pct","p95_delay_ms","device_energy_mj","compute_score","mean_reward","coverage_outage_observation_pct"]].round(5).to_string(index=False))
    print("VALIDATION="+str(out/"validation.json"))
if __name__=="__main__":main()
