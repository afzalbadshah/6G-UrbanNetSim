"""Bounded FR1/FR3 pilot using version 1.1 without changing simulator sources."""
from __future__ import annotations
import os,sys,json,time,hashlib,contextlib,copy
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace,MappingProxyType
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def state_hash(policy):
    return hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in policy.net.state_dict().values())).hexdigest()

def one_job(spec):
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    import torch
    torch.set_num_threads(1)
    import config as cfg
    import config.topology as topology
    from core.simulator import Simulator
    from scheduler.common.comparison_runner import ComparisonRunner
    from scheduler.common.policy_utils import set_global_seed
    home=Path(spec["home"]);home.mkdir(parents=True,exist_ok=True)
    os.chdir(home)
    base=topology.get_deployment_profile("fr1_city_coverage")
    profile=replace(base,profile_id="pilot_"+spec["band"].lower(),ground_bs_profile_id="nr-fr1-access" if spec["band"]=="FR1" else "nr-fr3-access",uav_profile_id="nr-fr1-air",access_generation="6G")
    topology.DEPLOYMENT_PROFILES=MappingProxyType({**topology.DEPLOYMENT_PROFILES,profile.profile_id:profile})
    cfg.SIMULATION["deployment_profile"]=profile.profile_id
    cfg.SIMULATION["interactive_deployment_profile"]=profile.profile_id
    report={"band":spec["band"],"model":spec["model"],"repetition":spec["index"],"phases":{}}
    with (home/"console.log").open("w",encoding="utf-8",buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        for training in (True,False):
            phase="train" if training else "evaluation"
            rounds=60 if training else 30
            seed=spec["train_seed"] if training else spec["eval_seed"]
            cfg.RANDOM_SEED=seed;set_global_seed(seed)
            workload=dict(cfg.apply_workload_profile(training,overrides={"start_devices":32,"max_devices":32,"increment":0,"rounds":rounds}))
            values={k:getattr(cfg,k) for k in dir(cfg) if k.isupper()}
            values.update(WRITE_DIAGNOSTICS_JSON=True,SMARTCITY_CSV_FLUSH_INTERVAL=200)
            runner=ComparisonRunner(spec["model"],config=SimpleNamespace(**values),output_dir=home/phase,seed=seed)
            checkpoint=home/"trained_model"/(spec["model"]+"_comparison_policy.pt")
            if not training and spec["model"]=="dqn":
                assert checkpoint.is_file(),checkpoint
                runner.load()
            runner.set_train_mode(training)
            before=state_hash(runner.policy) if spec["model"]=="dqn" else None
            sim=Simulator(runner,output_dir=home/phase,workload=workload,deployment_profile=profile.profile_id)
            handover=sim.mobility.handover_scheduler
            handover.model_dir=str(home/"trained_model/mobility")
            Path(handover.model_dir).mkdir(parents=True,exist_ok=True)
            if not training:
                assert (Path(handover.model_dir)/"handover_actor.pth").is_file()
            # Keep per-class workload definitions intact. Freeze the eMBB base
            # payload at 100 KB (device multipliers retained) for a stationary pilot.
            priorities=copy.deepcopy(cfg.PRIORITY_WORKLOADS)
            priorities[3].update(start_data_per_device_kb=100.0,data_per_device_kb=100.0,data_increment_kb=0.0)
            sim.generator.priority_workloads=priorities
            assert not any(n.kind.value=="ntn" for n in sim.physical_topology.nodes)
            layout=[(n.node_id,n.position.x_m,n.position.y_m,n.position.z_m) for n in sim.physical_topology.nodes]
            layout_digest=hashlib.sha256(json.dumps(sorted(layout)).encode()).hexdigest()
            device_mix=[(i,getattr(e,"device_type",None),float(e.x),float(e.y)) for i,e in enumerate(sim.mobility.entities)]
            node_profiles={n.node_id:n.profile_id for n in sim.physical_topology.nodes if n.kind.value in ("mec","access_tower","uav_bs")}
            started=time.monotonic()
            def progress(**event):
                if event.get("event")=="round_end":
                    rid=int(event["round_id"])
                    if rid%10==0 or rid==1:
                        current={"band":spec["band"],"model":spec["model"],"seed":seed,"phase":phase,"round":rid,"total_rounds":rounds,"elapsed_s":time.monotonic()-started}
                        (Path(spec["root"])/"current_job.json").write_text(json.dumps(current),encoding="utf-8")
            try:sim.run(train_mode=training,progress_callback=progress)
            finally:runner._close_csv()
            after=state_hash(runner.policy) if before else None
            if before:assert (before!=after) if training else (before==after)
            import pandas as pd
            file=home/phase/"offloading"/spec["model"]/"results.csv"
            d=pd.read_csv(file,sep=";",low_memory=False)
            assert len(d)==32*rounds and d.task_id.nunique()==len(d)
            assert set(d.class_name)=={"URLLC","mMTC","eMBB"}
            assert d.reward.notna().all()
            deadline_map={k:sorted(g.deadline_ms.unique().tolist()) for k,g in d.groupby("class_name")}
            assert deadline_map=={"URLLC":[100.0],"mMTC":[300.0],"eMBB":[500.0]},deadline_map
            from collections import Counter
            access_profiles=Counter()
            for execution in sim.realized_execution_by_task_id.values():
                network=getattr(execution,"network_execution",None)
                if network is not None and network.transmission.route.hops:
                    access_profiles[network.transmission.route.hops[0].hop.link_profile_id]+=1
            summary={"realized_access_profile_counts":dict(access_profiles),"seed":seed,"rounds":rounds,"devices":32,"decisions":len(d),"seconds":time.monotonic()-started,"source_csv":str(file),"layout_sha256":layout_digest,"device_mix":device_mix,"node_profiles":node_profiles,"deadline_map":deadline_map,"policy_before":before,"policy_after":after,"policy_diagnostics":getattr(runner.policy,"get_diagnostics",lambda:{})()}
            report["phases"][phase]=summary
            (home/"job_summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

def main():
    import config as cfg
    from config.radio import get_band
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out=ROOT/"output"/("aiot_channel_pilot_"+stamp)
    out.mkdir(parents=True,exist_ok=False)
    (ROOT/"output/latest_aiot_channel_pilot.txt").write_text(str(out),encoding="utf-8")
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ("config","core","scheduler","metrics","network","mobility","entities","workload") for p in (ROOT/folder).rglob("*.py")}
    jobs=[dict(root=str(out),home=str(out/band/("seed_"+str(i+1))/model),band=band,model=model,index=i+1,train_seed=42+i,eval_seed=1042+i) for i in range(2) for band in ("FR1","FR3") for model in ("random","dqn")]
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"simulator":str(ROOT),"source_hashes":hashes,"reward_profiles":cfg.PRIORITY_PROFILES,"bands":{b:dict(get_band(b)) for b in ("FR1","FR3")},"jobs":jobs,"design":{"ground_sites":81,"mec_compute_sites":cfg.SERVER_CONFIG["MEC"]["num_datacenters"],"uav_sites":cfg.SERVER_CONFIG["UAV_BS"]["num_datacenters"],"uav_access":"nr-fr1-air","lowband_anchor":"nr-lowband-access","area_km2":100,"train_rounds":60,"eval_rounds":30,"devices":32,"eMBB_base_kb":100,"common_application_deadlines_ms":{"URLLC":100,"mMTC":300,"eMBB":500},"notes":["Only ground-access band profile changes; profiles differ in frequency, bandwidth, fading, array gain, blockage and spatial streams. This is a channel-profile comparison, not an isolated carrier-frequency experiment.","FR1 radios retain their implementation's 5G link label; common 6G application deadlines are used in both cases to avoid changing QoS requirements.","Original multi-objective rewards, feasibility masks and spillover behavior retained.","Pilot uses short fresh training; it cannot establish convergence or production-scale capacity.","All offloading and mobility checkpoints are isolated under each job directory."]}}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("PILOT="+str(out),flush=True)
    results=[]
    with ProcessPoolExecutor(max_workers=1,max_tasks_per_child=1) as pool:
        for spec in jobs:
            result=pool.submit(one_job,spec).result()
            results.append(result)
            (out/"completed_jobs.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
            print(f"DONE {len(results)}/8 {spec['band']} seed_{spec['index']} {spec['model']}",flush=True)
    assert len({x["phases"]["train"]["layout_sha256"] for x in results})==1
    changed=[p for p,h in hashes.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    print("COMPLETE="+str(out),flush=True)
if __name__=="__main__":main()
