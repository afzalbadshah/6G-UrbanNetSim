"""Staged original-simulator training followed by evaluation seed 1 only."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil
ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8'); temp.replace(path)

def configuration_signature():
    import config as cfg
    files = ['config/reward.py','config/learning.py','config/workload.py','config/servers.py','config/radio.py','config/topology.py','config/simulation.py','scheduler/common/comparison_runner.py','scheduler/common/common_state.py','scheduler/common/common_action.py','scheduler/common/common_config.py']
    signature = {'deployment': cfg.SIMULATION['deployment_profile'],
            'configuration_hashes': {f: digest(ROOT/f) for f in files},
            'rounds': cfg.ROUNDS, 'max_devices': cfg.MAX_DEVICES,
            'workload_profile': cfg.WORKLOAD_PROFILE, 'workload_overrides': cfg.WORKLOAD_OVERRIDES}
    if cfg.ABLATE_HRLLC_LATENCY_SHAPING:
        signature['ablate_hrllc_latency_shaping'] = True
        signature['configuration_hashes']['metrics/reward.py'] = digest(ROOT/'metrics/reward.py')
    return signature

def prepare_job(job):
    import config as cfg
    model = job['model_name']
    if cfg.ABLATE_HRLLC_LATENCY_SHAPING and model != 'double_dqn':
        raise ValueError('HRLLC latency ablation is only supported for DDQN.')
    if job['train']:
        seed_root = Path(job['output_dir'])
    else:
        source = job.get('training_run')
        if not source:
            raise ValueError('Evaluation requires EVALUATION_TRAINING_RUN pointing to completed training.')
        seed_root = Path(source)/'exploration'/'seed_1'
        completion = seed_root/'offloading'/model/'checkpoint_manifest.json'
        manifest = json.loads(completion.read_text(encoding='utf-8'))
        if manifest['configuration'] != configuration_signature():
            raise ValueError('Evaluation configuration differs from training.')
        for filename, expected in manifest['checkpoint_hashes'].items():
            if digest(seed_root/filename) != expected:
                raise ValueError('Checkpoint hash mismatch: '+filename)
    return seed_root/'offloading'/model, seed_root/'mobility'/model

def finish_job(job, sim, checkpoint_root, mobility_root):
    import config as cfg
    import pandas as pd
    model=job['model_name']; output=Path(sim.output_dir)/'offloading'/model
    # Simulator output_dir is the per-model offloading directory in version 1.1.
    if not (output/'results.csv').is_file(): output=Path(sim.scheduler.output_dir)
    d=pd.read_csv(output/'results.csv', sep=';', usecols=['round','task_id','class_name','reward'])
    if len(d)!=cfg.decisions_per_model(sim.workload) or d.task_id.nunique()!=len(d):
        raise RuntimeError('Incomplete or duplicated task results: '+str(output))
    if d.reward.isna().any() or not set(d.class_name)=={'URLLC','mMTC','eMBB'}:
        raise RuntimeError(f'Invalid rewards or missing service classes: classes={set(d.class_name)}, missing_rewards={d.reward.isna().sum()}')
    seed_root=Path(job['output_dir']); files=[]
    if job['train']:
        # Include pending handover updates flushed by the simulator's final logs.
        sim.mobility.handover_scheduler.save()
        if model!='random': files.append(checkpoint_root/(model+'_comparison_policy.pt'))
        files += [mobility_root/'handover_actor.pth', mobility_root/'handover_critic.pth']
        hashes={str(p.relative_to(seed_root)):digest(p) for p in files}
        # Preserve legacy checkpoint filenames as well as the per-model snapshots.
        legacy=ROOT/'trained_model';legacy.mkdir(exist_ok=True)
        for p in files:
            if job.get("mirror_legacy", True) and not cfg.ABLATE_HRLLC_LATENCY_SHAPING:
                shutil.copy2(p,legacy/p.name)
        write_json(checkpoint_root/'checkpoint_manifest.json', {
            'configuration':configuration_signature(), 'checkpoint_hashes':hashes,
            'model':model,'training_seed':job['seed'],'decisions':len(d),
            'completed_utc':datetime.now(timezone.utc).isoformat()})
    write_json(output/'job_completion.json',{'model':model,'training':job['train'],
        'seed':job['seed'],'decisions':len(d),'rounds':int(d['round'].nunique()),
        'training_run':job.get('training_run'),'configuration':configuration_signature()})

def review_evaluation(run_dir):
    import pandas as pd
    from tools.analyze_aiot_channel_pilot import COLS, prepare, summarize
    phase=Path(run_dir)/'evaluation'; rows=[]
    import config as cfg
    for model in cfg.RUN_MODELS:
        p=phase/'seed_1'/'offloading'/model/'results.csv'
        d=prepare(pd.read_csv(p,sep=';',usecols=lambda c:c in COLS,low_memory=False))
        for cls,g in d.groupby('class_name'):
            row={'model':model,'task_class':cls,**summarize(g)}
            good=g[g.complete]
            row['completed_within_2ms_pct_of_attempts']=100*(g.complete & g.observed_complete_latency_ms.le(2)).mean()
            row['p99_delay_ms']=good.observed_complete_latency_ms.quantile(.99)
            rows.append(row)
    metrics=pd.DataFrame(rows);metrics.to_csv(phase/'first_seed_metrics.csv',index=False)
    # Keep three separate workload tables, with every measured objective retained.
    table_columns = {
        'URLLC':['model','attempts','deadline_success_pct','mean_delay_ms','p95_delay_ms','p99_delay_ms','mean_wireless_tx_ms','mean_wired_tx_ms','mean_propagation_ms','mean_queue_ms','mean_processing_ms','mean_handover_ms','device_energy_mj','battery_task_depletion_pct','headroom_pct','compute_score','mean_reward'],
        'mMTC':['model','attempts','deadline_success_pct','battery_task_depletion_pct','battery_remaining_pct','device_energy_mj','total_energy_j','mean_delay_ms','p95_delay_ms','p99_delay_ms','headroom_pct','compute_score','mean_reward'],
        'eMBB':['model','attempts','deadline_success_pct','headroom_pct','compute_score','success_adjusted_compute_score','task_goodput_mbps','mean_payload_kb','mean_delay_ms','p95_delay_ms','p99_delay_ms','device_energy_mj','battery_task_depletion_pct','mean_reward']}
    reward_description = (
        'DDQN uses the HRLLC latency-shaping ablation.'
        if cfg.ABLATE_HRLLC_LATENCY_SHAPING else 'All models use the original multi-objective rewards.'
    )
    report=['# Evaluation seed 1','',
        reward_description,
        'Battery task depletion covers recorded task battery draw; remaining battery also reflects other simulated battery use. Delay percentiles include completed tasks; deadline success uses all attempts.',
        'Capacity and headroom describe this 1-to-200 device ramp. High capacity scores alone do not establish performance beyond this load.',
        'The selected-action entropy metric in the CSV is empirical action diversity, rather than the neural policy entropy.', '']
    for cls,columns in table_columns.items():
        subset=metrics[metrics.task_class.eq(cls)]
        subset.to_csv(phase/(cls.lower()+'_model_comparison.csv'),index=False)
        t=subset[columns].round(5).fillna('NA')
        report += ['## '+cls,'','| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
        report += ['| '+' | '.join(str(value) for value in row)+' |' for row in t.itertuples(index=False,name=None)]
        report.append('')
    (phase/'first_seed_review.md').write_text('\n'.join(report),encoding='utf-8')
    write_json(phase/'review_status.json', {'status':'first_seed_complete_awaiting_review',
        'seeds_2_and_3_started':False,'metrics':str(phase/'first_seed_metrics.csv'),
        'note':'Compare delay, battery depletion and capacity alongside completion and all original objectives. One seed cannot establish statistical superiority.'})

def run_staged_campaign():
    import config as cfg
    from applications import MainApplication as app
    state=Path(app._configured_output_root())/'staged_campaign_status.json'
    report={'started_utc':datetime.now(timezone.utc).isoformat(),'status':'training',
            'evaluation_seeds_requested':1,'configuration':configuration_signature()}
    write_json(state,report)
    original=cfg.RUN_TRAINING
    try:
        app.run_configured_models()
        train_run=app.RUNTIME['run_dir']; report['training_run']=train_run
        if original and getattr(cfg,'AUTO_EVALUATE_FIRST_SEED',False):
            cfg.EVALUATION_TRAINING_RUN=train_run
            # Every model must have passed completion validation before evaluation.
            for model in cfg.RUN_MODELS:
                prepare_job({'model_name':model,'train':False,'training_run':train_run})
            report['status']='evaluation_seed_1';write_json(state,report)
            cfg.RUN_TRAINING=False
            app.run_configured_models()
            report['evaluation_run']=app.RUNTIME['run_dir']
            review_evaluation(report['evaluation_run'])
            report['status']='first_seed_complete_awaiting_review'
        else: report['status']='complete'
        report['completed_utc']=datetime.now(timezone.utc).isoformat();write_json(state,report)
    except BaseException as exc:
        report.update(status='failed',error=str(exc),run_dir=app.RUNTIME.get('run_dir'))
        write_json(state,report);raise
    finally:cfg.RUN_TRAINING=original
