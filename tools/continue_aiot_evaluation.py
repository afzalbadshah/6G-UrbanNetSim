"""Continue evaluation in an existing run without retraining or rewriting seed 1."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
import argparse, json, sys, shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def plan_jobs(training_run, evaluation_run):
    import config as cfg
    from config.evaluation import get_evaluation_plan, resolved_seeds
    from applications.staged_campaign import prepare_job, configuration_signature
    training_run=Path(training_run).resolve();evaluation_run=Path(evaluation_run).resolve()
    phase=evaluation_run/'evaluation'
    manifest=json.loads((phase/'manifest.json').read_text(encoding='utf-8-sig'))
    if manifest['mode']!='eval' or manifest['models']!=cfg.RUN_MODELS:
        raise ValueError('Existing evaluation models or mode differ from configured campaign')
    seeds=resolved_seeds(get_evaluation_plan(cfg.EVAL_PLAN_NAME),purpose='eval',count=3,master_seed=cfg.RANDOM_SEED)
    if manifest['seeds']['seed_1']!=seeds[0]:raise ValueError('Seed 1 derivation mismatch')
    for model in cfg.RUN_MODELS:
        done=json.loads((phase/'seed_1/offloading'/model/'job_completion.json').read_text())
        if done['training'] or done['seed']!=seeds[0] or done['configuration']!=configuration_signature():
            raise ValueError('Seed 1 configuration or completion mismatch: '+model)
        if Path(done['training_run']).resolve()!=training_run:
            raise ValueError('Training run differs from seed 1: '+model)
        prepare_job({'train':False,'model_name':model,'training_run':str(training_run)})
    jobs=[]
    for index in (1,2):
        destination=phase/f'seed_{index+1}'
        if destination.exists():
            raise FileExistsError('Refusing to overwrite existing seed directory: '+str(destination))
        for model in cfg.RUN_MODELS:
            jobs.append({'train':False,'model_name':model,'seed':seeds[index],
                'seed_index':index,'seed_total':3,'output_dir':str(destination),'training_run':str(training_run)})
    return phase,manifest,seeds,jobs

def build_metrics(phase, indices):
    import config as cfg
    import pandas as pd
    from tools.analyze_aiot_channel_pilot import COLS,prepare,summarize
    records=[]
    for index in indices:
        for model in cfg.RUN_MODELS:
            source=phase/f'seed_{index}'/'offloading'/model/'results.csv'
            frame=prepare(pd.read_csv(source,sep=';',usecols=lambda c:c in COLS,low_memory=False))
            for cls,g in frame.groupby('class_name'):
                record={'seed_index':index,'model':model,'task_class':cls,**summarize(g)}
                record['completed_within_2ms_pct_of_attempts']=100*(g.complete & g.observed_complete_latency_ms.le(2)).mean()
                record['p99_delay_ms']=g.loc[g.complete,'observed_complete_latency_ms'].quantile(.99)
                records.append(record)
    table=pd.DataFrame(records)
    table.to_csv(phase/'evaluation_metrics_by_seed.csv',index=False)
    for index in indices:
        if index!=1:table[table.seed_index.eq(index)].to_csv(phase/f'seed_{index}'/'class_metrics.csv',index=False)
    if len(indices)==3:
        numeric=[c for c in table.select_dtypes(include='number').columns if c!='seed_index']
        agg=table.groupby(['model','task_class'])[numeric].agg(['mean','std'])
        agg.columns=[a+'_'+b for a,b in agg.columns]
        agg.to_csv(phase/'evaluation_metrics_aggregate.csv')

def run(training_run,evaluation_run):
    from applications.MainApplication import _run_one_job
    from applications.staged_campaign import write_json,digest
    phase,manifest,seeds,jobs=plan_jobs(training_run,evaluation_run)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup=ROOT/'.backups'/('before_eval_seeds_2_3_'+stamp);backup.mkdir(parents=True)
    root_state=ROOT/'output/staged_campaign_status.json'
    for p in (phase/'manifest.json',phase/'review_status.json',root_state):
        if p.is_file():shutil.copy2(p,backup/p.name)
    seed1_hashes={str(p.relative_to(phase)):digest(p) for p in (phase/'seed_1').rglob('*') if p.is_file()}
    write_json(backup/'seed_1_hashes.json',seed1_hashes)
    state=json.loads(root_state.read_text(encoding='utf-8-sig'))
    state.update(status='evaluation_seeds_2_3',evaluation_run=str(Path(evaluation_run).resolve()),
        evaluation_seeds_requested=3,completed_evaluation_seeds=[1],continuation_started_utc=datetime.now(timezone.utc).isoformat())
    state.pop('completed_utc',None)
    manifest.update(seeds={f'seed_{i+1}':value for i,value in enumerate(seeds)},repetitions=3,
        completed_seed_indices=[1],continuation_started_utc=state['continuation_started_utc'])
    write_json(phase/'manifest.json',manifest);write_json(root_state,state)
    write_json(phase/'review_status.json',{'status':'evaluation_seeds_2_3_running','seeds_2_and_3_started':True,'completed_seed_indices':[1]})
    print('[CONTINUE] Existing evaluation:',phase,flush=True)
    print('[SEEDS]',dict(enumerate(seeds,1)),flush=True)
    try:
        for job in jobs:
            state.update(current_seed_index=job['seed_index']+1,current_model=job['model_name'])
            write_json(root_state,state)
            # One fresh worker per model releases memory and keeps the load bounded.
            with ProcessPoolExecutor(max_workers=1) as pool:result=pool.submit(_run_one_job,job).result()
            print('[DONE]',result,flush=True)
            if job is jobs[-1] or jobs[jobs.index(job)+1]['seed_index']!=job['seed_index']:
                index=job['seed_index']+1
                state['completed_evaluation_seeds'].append(index)
                manifest['completed_seed_indices']=state['completed_evaluation_seeds']
                write_json(phase/'manifest.json',manifest);write_json(root_state,state)
                build_metrics(phase,state['completed_evaluation_seeds'])
        for filename,expected in seed1_hashes.items():
            if digest(phase/filename)!=expected:raise RuntimeError('Seed 1 changed: '+filename)
        # Evaluation must also leave all training checkpoint hashes intact.
        from applications.staged_campaign import prepare_job
        for job in jobs:prepare_job(job)
        state.update(status='three_seed_evaluation_complete_awaiting_review',completed_utc=datetime.now(timezone.utc).isoformat(),seed_1_unchanged=True)
        write_json(root_state,state)
        write_json(phase/'review_status.json',{'status':state['status'],'completed_seed_indices':[1,2,3],
            'seeds_2_and_3_started':True,'seed_1_unchanged':True,'training_checkpoints_unchanged':True,
            'per_seed_metrics':str(phase/'evaluation_metrics_by_seed.csv'),
            'aggregate_metrics':str(phase/'evaluation_metrics_aggregate.csv'),
            'aggregation_note':'Means and sample standard deviations across three seed-level metrics; mean percentiles are not pooled percentiles.'})
        print('[COMPLETE] Evaluation seeds 1, 2 and 3 complete in',phase,flush=True)
    except BaseException as exc:
        state.update(status='evaluation_seeds_2_3_failed',error=str(exc));write_json(root_state,state);raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--training-run',required=True);parser.add_argument('--evaluation-run',required=True);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.check_only:
        phase,manifest,seeds,jobs=plan_jobs(args.training_run,args.evaluation_run)
        print(json.dumps({'phase':str(phase),'seeds':dict(enumerate(seeds,1)),'new_jobs':len(jobs),
            'rounds_per_job':200,'decisions_per_job':20100,'training':False,'seed_1_will_be_rewritten':False},indent=2))
    else:
        from applications.system_awake import run_while_system_awake
        run_while_system_awake(lambda:run(args.training_run,args.evaluation_run),logger=lambda message:print(message,flush=True))
if __name__=='__main__':main()
