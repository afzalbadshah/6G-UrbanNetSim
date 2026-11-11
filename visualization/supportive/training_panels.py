"""Training-only diagnostics resolved from the selected evaluation's provenance.

Internal helper for delay_plotter and energy_plotter, not a third entry point.
No evaluation observations enter training metrics; one selected training seed.
"""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from visualization.paper_plotting import MODELS,LABELS,COLORS,STYLES,ROOT,save_panel

COLUMNS=['round','task_id','reward','action_entropy','success','status',
         'observed_complete_latency_ms','deadline_ms','device_battery_energy_j']
NEURAL=('dqn','double_dqn','ppo','a2c')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def resolve_training(phase,seeds,training_seed_index=None):
    phase=Path(phase).resolve()
    evaluation_manifest=read_json(phase/'manifest.json')
    if evaluation_manifest.get('mode')!='eval':raise ValueError('Expected an evaluation phase')
    roots=set();jobs=[]
    for seed in seeds:
        for model in MODELS:
            path=phase/f'seed_{seed}'/'offloading'/model/'job_completion.json'
            job=read_json(path)
            if job.get('training') is not False or not job.get('training_run'):
                raise ValueError(f'Missing evaluation-to-training provenance: {path}')
            if job.get('model')!=model or job.get('seed')!=evaluation_manifest['seeds'][f'seed_{seed}']:
                raise ValueError(f'Wrong model or seed: {path}')
            roots.add(Path(job['training_run']).resolve());jobs.append((path,job))
    if len(roots)!=1:raise ValueError('Evaluation jobs reference different training runs')
    run=roots.pop();training_phase=run/'exploration';manifest=read_json(training_phase/'manifest.json')
    if manifest.get('mode')!='train':raise ValueError('Linked source is not training')
    choices=list(manifest['seeds'])
    if training_seed_index is None:
        if len(choices)!=1:raise ValueError('Multiple training seeds: select --training-seed-index explicitly')
        seed_folder=choices[0]
    else:seed_folder=f'seed_{training_seed_index}'
    if seed_folder not in choices:raise ValueError('Requested training seed is absent')
    home=training_phase/seed_folder
    for _,job in jobs:
        train_job=read_json(home/'offloading'/job['model']/'job_completion.json')
        if train_job.get('configuration')!=job.get('configuration'):
            raise ValueError('Training/evaluation configuration mismatch')
    return home,manifest,[{'path':str(p),'sha256':digest(p)} for p,_ in jobs]


def summarise_rounds(data):
    complete=data.status.eq('completed')
    timely=complete & data.observed_complete_latency_ms.le(data.deadline_ms)
    if not np.array_equal(timely.to_numpy(),data.success.eq(1).to_numpy()):
        raise ValueError('Training success flags disagree with observed deadlines')
    d=data.assign(on_time=timely,energy_mj=1000*data.device_battery_energy_j,
                  completed_latency=data.observed_complete_latency_ms.where(complete))
    return d.groupby('round').agg(attempts=('task_id','size'),mean_reward=('reward','mean'),
        empirical_action_entropy_nats=('action_entropy','mean'),deadline_success=('on_time','mean'),
        device_energy_mj_per_attempt=('energy_mj','mean'),
        completed_latency_ms=('completed_latency','mean')).reset_index().assign(
            deadline_success_pct=lambda x:100*x.deadline_success)


def load_training(home,manifest):
    summaries=[];losses=[];sources=[]
    for model in MODELS:
        folder=home/'offloading'/model;jobpath=folder/'job_completion.json';job=read_json(jobpath)
        checkpoint_path=folder/'checkpoint_manifest.json';checkpoint=read_json(checkpoint_path)
        if job.get('training') is not True or job.get('model')!=model:
            raise ValueError(f'Wrong training source: {folder}')
        if job['seed']!=manifest['seeds'][home.name] or checkpoint['training_seed']!=job['seed']:
            raise ValueError('Training seed mismatch')
        if checkpoint['configuration']!=job['configuration'] or checkpoint['decisions']!=job['decisions']:
            raise ValueError('Checkpoint metadata mismatch')
        for relative,expected in checkpoint['checkpoint_hashes'].items():
            artifact=home/Path(relative.replace('\\','/'))
            if digest(artifact)!=expected:raise ValueError(f'Checkpoint has changed: {artifact}')
        source=folder/'results.csv';data=pd.read_csv(source,sep=';',usecols=COLUMNS,low_memory=False)
        if len(data)!=job['decisions'] or data.task_id.nunique()!=len(data):raise ValueError('Partial or duplicated training tasks')
        if set(data['round'])!=set(range(1,job['rounds']+1)):raise ValueError('Incomplete training rounds')
        numeric=['round','reward','action_entropy','success','device_battery_energy_j','deadline_ms']
        if not np.isfinite(data[numeric].to_numpy()).all():raise ValueError('Nonfinite training metric')
        if (data.device_battery_energy_j<0).any() or (data.action_entropy< -1e-9).any():raise ValueError('Negative energy or entropy')
        completed=data.status.eq('completed')
        if not np.isfinite(data.loc[completed,'observed_complete_latency_ms']).all():raise ValueError('Missing completion latency')
        summary=summarise_rounds(data);summary['model']=model;summaries.append(summary)
        for path in [source,jobpath,checkpoint_path]:sources.append({'model':model,'path':str(path),'sha256':digest(path)})
        lossfile=folder/(model+'_loss_log.csv')
        if model in NEURAL:
            loss=pd.read_csv(lossfile)
            if loss.empty or loss['round'].duplicated().any() or not np.isfinite(loss[['round','loss']]).all().all():
                raise ValueError('Invalid training loss log')
            if not set(loss['round']).issubset(set(summary['round'])):raise ValueError('Loss recorded outside training rounds')
            loss['model']=model;losses.append(loss);sources.append({'model':model,'path':str(lossfile),'sha256':digest(lossfile)})
    return pd.concat(summaries,ignore_index=True),pd.concat(losses,ignore_index=True),sources


def learning_parameters(home):
    checkpoint=read_json(home/'offloading/dqn/checkpoint_manifest.json')
    path=ROOT/'config/learning.py';expected=checkpoint['configuration']['configuration_hashes'].get('config/learning.py')
    if digest(path)!=expected:return {'status':'Current configuration differs from training snapshot; batch settings not inferred.'}
    names={'BATCH_SIZE','LEARN_EVERY_N_DECISIONS','PPO_EPOCHS'};values={}
    for node in ast.parse(path.read_text(encoding='utf-8-sig')).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in names:values[target.id]=ast.literal_eval(node.value)
    return values


def draw_metric(summary,metric,ylabel,target,window=5):
    fig,ax=plt.subplots(figsize=(4.4,3.5),layout='constrained')
    for model in MODELS:
        d=summary[summary.model.eq(model)].sort_values('round')
        ax.plot(d['round'],d[metric],color=COLORS[model],alpha=.18,linewidth=.55)
        ax.plot(d['round'],d[metric].rolling(window,min_periods=1).mean(),color=COLORS[model],
            linestyle=STYLES[model],linewidth=1.25,label=LABELS[model]+(' baseline' if model=='random' else ''))
    ax.set_xlabel('Training round');ax.set_ylabel(ylabel);ax.set_xlim(1,int(summary['round'].max()))
    ax.legend(loc='best',ncol=2,fontsize=6.3,framealpha=.9)
    if metric=='deadline_success_pct':ax.set_ylim(0,100.5)
    if metric in ['empirical_action_entropy_nats','device_energy_mj_per_attempt']:ax.set_ylim(bottom=0)
    save_panel(fig,target);plt.close(fig)


def draw_loss(losses,target):
    fig,ax=plt.subplots(figsize=(4.4,3.5),layout='constrained')
    for model in NEURAL:
        d=losses[losses.model.eq(model)].sort_values('round')
        ax.plot(d['round'],d.loss,color=COLORS[model],linestyle=STYLES[model],linewidth=1,label=LABELS[model])
    ax.set_xlabel('Training round');ax.set_ylabel('Last reported update loss (symlog scale)')
    ax.set_yscale('symlog',linthresh=.1);ax.axhline(0,color='#999999',linewidth=.5)
    ax.set_xlim(1,int(losses['round'].max()));ax.legend(loc='best',ncol=2,fontsize=6.5)
    save_panel(fig,target);plt.close(fig)


def render_training(phase,seeds,output,direction,training_seed_index=None,validate_only=False):
    home,manifest,evaluation_jobs=resolve_training(phase,seeds,training_seed_index)
    summary,losses,sources=load_training(home,manifest)
    if validate_only:return
    output=Path(output)/'training';output.mkdir(parents=True,exist_ok=True)
    data_dir=output/'data';data_dir.mkdir(exist_ok=True)
    specs=[('Training reward.png','mean_reward','Mean reward per attempted task'),
           ('Training empirical action entropy.png','empirical_action_entropy_nats','Mean empirical action entropy (nats)')]
    if direction=='delay':specs.append(('Training deadline success.png','deadline_success_pct','Training deadline success (%)'))
    else:specs.append(('Training device energy.png','device_energy_mj_per_attempt','Mean device task energy (mJ/attempt)'))
    for name,metric,label in specs:draw_metric(summary,metric,label,output/name)
    draw_loss(losses,output/'Training update loss.png')
    summary.to_csv(data_dir/'training_round_metrics.csv',index=False)
    losses.to_csv(data_dir/'training_last_update_loss.csv',index=False)
    record={'training_seed_folder':home.name,'training_seed':manifest['seeds'][home.name],
        'training_source':str(home),'evaluation_phase':str(Path(phase).resolve()),'evaluation_seeds':list(seeds),
        'sources':sources,'evaluation_link_records':evaluation_jobs,'learning_parameters':learning_parameters(home),
        'rounds':int(summary['round'].max()),'models':list(MODELS),'attempts_per_model':summary.groupby('model').attempts.sum().to_dict(),
        'export':'four separate PNGs, 300 dpi, no headings, local legends',
        'smoothing':'Reward, entropy and service metrics: raw round mean faint; trailing 5-round arithmetic mean prominent. Loss is unsmoothed.',
        'entropy':'Mean of logged causal per-class last-50-executed-action Shannon entropies, natural logarithm; not neural policy entropy.',
        'loss':'Last available learn() loss per round, not all batch losses. Different model objectives; magnitudes are not comparable across algorithms. Q-learning and Random have no neural loss.',
        'fourth_panel':'Draft for selection: deadline success for delay, task-attributed device energy per attempt for energy. Does not isolate battery lifetime.',
        'limits':['One training seed, not three independent training repetitions. Evaluation seeds reuse these checkpoints.',
                  'Workload changes during training. Trends do not isolate learning from increasing load or establish convergence.',
                  'Shading is not used for uncertainty; faint traces are raw values.',
                  'Random is a nonlearning baseline. No separate handover actor training is mixed into these offloading plots.'],
        'files':[x[0] for x in specs]+['Training update loss.png']}
    (output/'training_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    (output/'README.md').write_text('# Linked training diagnostics\n\n'+
        '\n\n'.join([f'Source: {home}',record['smoothing'],record['entropy'],record['loss'],record['fourth_panel'],*record['limits']]),encoding='utf-8')
    print(f'[TRAINING] Wrote four panels to {output} from {home}',flush=True)
