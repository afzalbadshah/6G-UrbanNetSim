"""Shared data, metrics and IEEE rendering for the two paper plotters.

All metrics use generated task attempts. Workload is offered Gcycles/round,
not device count or unverified tasks/s. Environmental seed ranges are not CIs.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_PHASE=ROOT/'output/run_002_fr3_urban_aiot_evaluation/evaluation'
MODELS=('dqn','double_dqn','ppo','a2c','q_learning','random')
LABELS={'dqn':'DQN','double_dqn':'DDQN','ppo':'PPO','a2c':'A2C','q_learning':'Q-learning','random':'Random'}
COLORS=dict(zip(MODELS,['#D62728','#2CA02C','#1F77B4','#FF7F0E','#9467BD','#222222']))
STYLES=dict(zip(MODELS,['-','--','-.',':',(0,(4,1,1,1)),(0,(1,1))]))
CLASSES=('URLLC','mMTC','eMBB')
TEXT={'task_id','device_type','class_name','status','network_failure_reason','execution_tier'}
COLS=list(TEXT)+['round','device_id','success','deadline_ms','observed_complete_latency_ms',
 'active_devices_this_round','compute_work_cycles','input_payload_bytes','device_battery_energy_j',
 'battery_capacity_j','device_battery_before_j','device_battery_after_j','device_tx_energy_j',
 'device_rx_energy_j','device_local_compute_energy_j','device_wait_energy_j','device_propulsion_energy_j',
 'infra_compute_energy_j','backhaul_energy_j','total_energy_j','tx_delay_wireless_ms','tx_delay_wired_ms',
 'propagation_total_ms','queue_delay_ms','processing_delay_ms','handover_delay_component_ms']

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
 'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,
 'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
 'axes.grid':True,'grid.alpha':.22,'grid.linewidth':.4,'axes.axisbelow':True,
 'savefig.facecolor':'white','figure.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42})

def energy_per_useful(energy,success):
    count=int(np.asarray(success,dtype=bool).sum())
    return float(np.asarray(energy,dtype=float).sum()/count) if count else np.nan

def longest_miss_streak(success,rounds):
    longest=run=0;previous=None
    for good,rid in zip(success,rounds):
        if previous is not None and rid!=previous+1:run=0
        run=0 if good else run+1
        longest=max(longest,run);previous=rid
    return longest

def load_evaluation(phase,seeds):
    phase=Path(phase).resolve()
    manifest=json.loads((phase/'manifest.json').read_text(encoding='utf-8-sig'))
    if manifest.get('mode')!='eval':raise ValueError('Source must be an evaluation phase')
    frames=[];sources=[]
    for seed in seeds:
        if f'seed_{seed}' not in manifest['seeds']:raise ValueError(f'Seed {seed} is absent from manifest')
        for model in MODELS:
            home=phase/f'seed_{seed}'/'offloading'/model
            job=json.loads((home/'job_completion.json').read_text(encoding='utf-8-sig'))
            if job['training'] or int(job['seed'])!=int(manifest['seeds'][f'seed_{seed}']):
                raise ValueError(f'Wrong phase or seed: {home}')
            source=home/'results.csv'
            d=pd.read_csv(source,sep=';',usecols=lambda c:c in COLS,low_memory=False)
            required=set(COLS)-{'device_propulsion_energy_j'}
            if required-set(d.columns):raise ValueError(f'Missing columns in {source}: {required-set(d.columns)}')
            if len(d)!=job['decisions'] or d.task_id.nunique()!=len(d) or d['round'].nunique()!=job['rounds']:
                raise ValueError(f'Incomplete or duplicated results: {source}')
            for c in set(d)-TEXT:d[c]=pd.to_numeric(d[c],errors='coerce')
            for c in ['round','device_id','deadline_ms','compute_work_cycles','input_payload_bytes','device_battery_energy_j','battery_capacity_j']:
                if not np.isfinite(d[c]).all() or (d[c]<0).any():raise ValueError(f'Invalid {c}: {source}')
            if not set(d.class_name)==set(CLASSES):raise ValueError(f'Missing service classes: {source}')
            d['complete']=d.status.eq('completed')
            d['on_time']=d.complete & d.observed_complete_latency_ms.le(d.deadline_ms)
            if not np.array_equal(d.on_time.to_numpy(),d.success.eq(1).to_numpy()):
                raise ValueError(f'Success flag disagrees with observed completion/deadline: {source}')
            d['late']=d.complete & ~d.on_time;d['failed']=~d.complete
            d['radio_ms']=d.tx_delay_wireless_ms+d.tx_delay_wired_ms+d.propagation_total_ms
            d['radio_energy_j']=d.device_tx_energy_j+d.device_rx_energy_j
            d['infra_energy_j']=d.infra_compute_energy_j+d.backhaul_energy_j
            if 'device_propulsion_energy_j' in d and d.device_propulsion_energy_j.isna().any():
                raise ValueError('Invalid propulsion energy')
            d['model']=model;d['seed']=seed
            loads=d.groupby('round')[['compute_work_cycles','input_payload_bytes']].sum()
            d['workload']=d['round'].map(loads.compute_work_cycles/1e9)
            d['offered_mb']=d['round'].map(loads.input_payload_bytes/1e6)
            frames.append(d)
            sources.append({'seed_index':seed,'model':model,'path':str(source),
                'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'attempts':len(d)})
    data=pd.concat(frames,ignore_index=True)
    return data,sources

class Context:
    def __init__(self,phase,seeds,output):
        self.phase=Path(phase).resolve();self.seeds=tuple(seeds);self.output=Path(output).resolve()
        self.data,self.sources=load_evaluation(self.phase,self.seeds)
        self.devices=self._device_metrics()
        lo,hi=self.data.workload.min(),self.data.workload.max()
        self.edges=np.linspace(lo,hi if hi>lo else lo+1,11)
        self.records=[];self.tables={}
    def _device_metrics(self):
        rows=[]
        for (model,seed,device,cls),g in self.data.groupby(['model','seed','device_id','class_name'],sort=False):
            g=g.sort_values('round');cap=g.battery_capacity_j.iloc[0]
            if not np.isfinite(cap) or cap<=0:continue
            first=g.device_battery_before_j.iloc[0];last=g.device_battery_after_j.iloc[-1]
            rows.append({'model':model,'seed':seed,'device_id':device,'class_name':cls,
                'depletion':100*(first-last)/cap,'remaining':100*last/cap,'initial':100*first/cap,
                'success_pct':100*g.on_time.mean(),'attempts':len(g),'active_rounds':g['round'].nunique(),
                'longest_miss':longest_miss_streak(g.on_time.to_numpy(),g['round'].to_numpy())})
        return pd.DataFrame(rows)
    def subset(self,model,cls=None,complete=False):
        d=self.data[self.data.model.eq(model)]
        if cls is not None:d=d[d.class_name.eq(cls)]
        return d[d.complete] if complete else d
    def add_table(self,code,table):self.tables[code]=table

@dataclass
class Panel:
    code:str
    title:str
    family:str
    draw:object=None
    pending:str=''

def _ticks(ax):
    ax.set_xticks(range(len(MODELS)),[LABELS[m] for m in MODELS],rotation=32,ha='right')

def line_range(ax,x,values,model):
    arr=np.asarray(values,dtype=float);valid=np.isfinite(arr).any(axis=0)
    mean=low=high=None
    if not valid.any():return
    mean=np.full(len(x),np.nan);low=mean.copy();high=mean.copy()
    mean[valid]=np.nanmean(arr[:,valid],axis=0);low[valid]=np.nanmin(arr[:,valid],axis=0);high[valid]=np.nanmax(arr[:,valid],axis=0)
    ax.fill_between(x,low,high,color=COLORS[model],alpha=.12,linewidth=0)
    ax.plot(x,mean,color=COLORS[model],linestyle=STYLES[model],linewidth=1.2)

def workload_curve(ctx,ax,code,cls=None,metric='on_time',scale=100,ylabel='Deadline success (%)'):
    rows=[];centers=(ctx.edges[:-1]+ctx.edges[1:])/2
    for model in MODELS:
        series=[]
        for seed in ctx.seeds:
            d=ctx.subset(model,cls);d=d[d.seed.eq(seed)].copy()
            d['bin']=pd.cut(d.workload,ctx.edges,labels=False,include_lowest=True)
            g=d.groupby('bin',observed=True)[metric].agg(['mean','count']).reindex(range(10))
            vals=g['mean'].to_numpy()*scale;series.append(vals)
            for i in range(10):rows.append({'model':model,'seed':seed,'workload_gcycles_per_round':centers[i],
                'value':vals[i],'attempts':g['count'].iloc[i]})
        line_range(ax,centers,series,model)
    ax.set_xlabel('Offered workload (Gcycles/round)');ax.set_ylabel(ylabel)
    finite=np.array([r['value'] for r in rows],dtype=float)
    finite=finite[np.isfinite(finite)]
    if metric=='on_time' and len(finite):
        low=max(0,float(finite.min())-max(.05,(100-finite.min())*.12))
        ax.set_ylim(low,100.05)
    elif metric=='late' and len(finite):
        ax.set_ylim(0,max(1,float(finite.max())*1.12))
    ctx.add_table(code,pd.DataFrame(rows))

def matrix(ctx,ax,code,fn,label,device=False,fmt='.2f',scale=1,cmap='viridis'):
    rows=[];table=np.full((6,3),np.nan)
    for i,model in enumerate(MODELS):
        for j,cls in enumerate(CLASSES):
            vals=[]
            for seed in ctx.seeds:
                d=ctx.devices if device else ctx.data
                g=d[d.model.eq(model)&d.seed.eq(seed)&d.class_name.eq(cls)]
                val=float(fn(g))*scale if len(g) else np.nan;vals.append(val)
                rows.append({'model':model,'seed':seed,'class_name':cls,'value':val,'observations':len(g)})
            if np.isfinite(vals).any():table[i,j]=np.nanmean(vals)
    finite=table[np.isfinite(table)]
    vmin=float(finite.min()) if len(finite) else 0;vmax=float(finite.max()) if len(finite) else 1
    if vmax<=vmin:vmax=vmin+1
    im=ax.imshow(np.ma.masked_invalid(table),aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax)
    for i in range(6):
        for j in range(3):
            val=table[i,j];text=format(val,fmt) if np.isfinite(val) else 'N/A'
            ax.text(j,i,text,ha='center',va='center',fontsize=7,
                color='white' if np.isfinite(val) and (val-vmin)/(vmax-vmin)<.4 else 'black')
    ax.set_xticks(range(3),CLASSES);ax.set_yticks(range(6),[LABELS[m] for m in MODELS]);ax.grid(False)
    bar=ax.figure.colorbar(im,ax=ax,pad=.025,fraction=.045);bar.set_label(label,fontsize=7)
    ctx.add_table(code,pd.DataFrame(rows))

def distribution(ctx,ax,code,cls,column,device=False,ccdf=False):
    base=ctx.devices if device else ctx.data[ctx.data.complete]
    base=base[base.class_name.eq(cls)];v=pd.to_numeric(base[column],errors='coerce').dropna()
    if v.empty:raise ValueError('No observations: '+code)
    if ccdf:
        pos=v[v>0];grid=np.unique(np.r_[0,np.geomspace(max(pos.min()*.95,1e-9),pos.max()*1.01,450)])
    else:grid=np.linspace(min(0,v.min()),v.max()*1.01+1e-12,450)
    rows=[]
    for model in MODELS:
        curves=[]
        for seed in ctx.seeds:
            a=np.sort(base.loc[base.model.eq(model)&base.seed.eq(seed),column].dropna().to_numpy())
            if len(a)==0:raise ValueError(f'No observations for {model}, seed {seed}, panel {code}')
            y=np.searchsorted(a,grid,side='right')/len(a);y=1-y if ccdf else y
            curves.append(y)
            rows += [{'model':model,'seed':seed,'x':x,'probability':q,'observations':len(a)} for x,q in zip(grid,y)]
        line_range(ax,grid,curves,model)
    ax.set_ylabel('P(completion latency > x)' if ccdf else 'Cumulative fraction of devices')
    if ccdf:
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(max(v[v>0].min()*.95,1e-9),v.max()*1.01)
        ax.set_ylim(max(1/base.groupby(['model','seed']).size().max(),1e-6)*.7,1.05)
        ax.set_xlabel('Completion latency (ms)')
    else:
        ax.set_xlabel('Battery depletion (percentage points)');ax.set_ylim(0,1.02)
        ax.set_xscale('symlog',linthresh=.01)
    ctx.add_table(code,pd.DataFrame(rows))

def model_bars(ctx,ax,code,fn,label,cls=None):
    rows=[]
    for i,model in enumerate(MODELS):
        vals=[]
        for seed in ctx.seeds:
            g=ctx.subset(model,cls);g=g[g.seed.eq(seed)];value=float(fn(g));vals.append(value)
            rows.append({'model':model,'seed':seed,'value':value,'attempts':len(g)})
        a=np.asarray(vals);valid=a[np.isfinite(a)]
        if not len(valid):continue
        mean=valid.mean();ax.bar(i,mean,color=COLORS[model],alpha=.85,width=.65)
        ax.errorbar(i,mean,yerr=[[mean-valid.min()],[valid.max()-mean]],color='#444444',capsize=2,linewidth=.7)
        ax.scatter(np.full(len(valid),i),valid,color='#333333',s=7,zorder=4)
    _ticks(ax);ax.set_ylabel(label);ax.set_ylim(bottom=0)
    ax.legend(handles=[Patch(facecolor='#aaaaaa',label='Seed mean'),
        Line2D([],[],color='#333333',marker='o',linestyle='none',markersize=3,label='Individual seed'),
        Line2D([],[],color='#444444',marker='|',label='Seed min–max')],loc='best',fontsize=6.5)
    ctx.add_table(code,pd.DataFrame(rows))

def pending_tile(ax,panel):
    ax.set_axis_off();ax.text(.5,.58,'PENDING EXPERIMENT',ha='center',va='center',fontsize=9,color='#555555',transform=ax.transAxes)
    import textwrap
    ax.text(.5,.37,'\n'.join(textwrap.wrap(panel.pending,43)),ha='center',va='center',fontsize=8,color='#666666',transform=ax.transAxes)

def panel_filename(title):
    import re
    stem=re.sub(r'[<>:"/\\|?*\x00-\x1f]+',' - ',title)
    stem=re.sub(r'\s+',' ',stem).strip(' .')
    if not stem:raise ValueError('Panel title must produce a nonempty filename')
    return stem+'.png'


def save_panel(fig,target):
    import os,tempfile,time
    with tempfile.NamedTemporaryFile(prefix='export_',suffix='.png',dir=target.parent,delete=False) as temp:
        temporary=Path(temp.name)
    try:
        fig.savefig(temporary,dpi=300,format='png')
        for attempt in range(4):
            try:
                os.replace(temporary,target);break
            except OSError:
                if attempt==3:raise
                time.sleep(.5*(attempt+1))
    finally:
        temporary.unlink(missing_ok=True)


def render(ctx,groups,direction):
    ctx.output.mkdir(parents=True,exist_ok=True);(ctx.output/'data').mkdir(exist_ok=True)
    (ctx.output/'run_status.json').write_text(json.dumps({'status':'running','seeds':list(ctx.seeds)}),encoding='utf-8')
    used_names=set()
    for group_name,panels in groups:
        for panel in panels:
            filename=panel_filename(panel.title) if panel.draw else None
            if filename in used_names:raise ValueError('Duplicate panel filename: '+filename)
            if filename:used_names.add(filename)
            ctx.records.append({'id':panel.code,'title':panel.title,'plot_type':panel.family,
                'status':'generated' if panel.draw else 'pending','reason':panel.pending or None,
                'group':group_name,'filename':filename})
            if panel.draw is None:continue
            fig,ax=plt.subplots(figsize=(4.4,3.5),layout='constrained')
            panel.draw(ctx,ax,panel.code)
            # Semantic component legends and heatmap colour scales remain local.
            model_legend=any(token in panel.family for token in ('line','ECDF','CCDF','scatter'))
            if model_legend:
                existing=ax.get_legend()
                if existing is not None:
                    handles=[Line2D([],[],color=COLORS[m],marker='o',linestyle='none',label=LABELS[m]) for m in MODELS]
                    old_handles=getattr(existing,'legend_handles',getattr(existing,'legendHandles',[]))
                    handles+=list(old_handles)
                else:
                    handles=[Line2D([],[],color=COLORS[m],linestyle=STYLES[m],label=LABELS[m],linewidth=1.5) for m in MODELS]
                ax.legend(handles=handles,loc='best',ncol=2,fontsize=6.5,
                    frameon=True,framealpha=.9,edgecolor='#dddddd')
            # The descriptive filename replaces all panel and figure headings.
            ax.set_title('');fig.suptitle('')
            save_panel(fig,ctx.output/filename)
            plt.close(fig);print('[WROTE]',filename,flush=True)
    for code,table in ctx.tables.items():table.to_csv(ctx.output/'data'/(code+'.csv'),index=False)
    provenance={'created_utc':datetime.now(timezone.utc).isoformat(),'direction':direction,
        'evaluation_phase':str(ctx.phase),'seeds':list(ctx.seeds),'models':list(MODELS),'sources':ctx.sources,
        'workload':'Offered computation summed over all generated task attempts per round, divided by 1e9',
        'interpretation':'Existing device count and eMBB payload grow together: coupled workload ramp, not an independent workload sweep.',
        'latency_population':'Completed tasks only; outcome figures retain all generated attempts.',
        'axes':'Success line charts use explicitly labelled restricted y ranges for visibility; bar charts start at zero. Battery ECDF uses symlog x with 0.01 percentage-point linear threshold.',
        'uncertainty':'Environmental seed mean and observed min–max; not a confidence interval or independent training variation.',
        'battery_population':'Battery distributions/charge exclude mains-powered records with nonpositive rated battery capacity. Same device identities and exposure across models within each seed; active durations vary within a seed.',
        'export':{'format':'png','layout':'one measured panel per file','titles':False,'resolution_dpi':300},
        'panels':ctx.records}
    (ctx.output/'plot_manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    notes=['# '+direction.title()+' plots','',f'Source: {ctx.phase}',f'Seeds: {ctx.seeds}',
        '','The figure plan has 24 panel slots. Each measured panel is exported as its own PNG, without a heading; its filename gives the panel title.',
        'Pending panels are listed here but are not exported. No synthetic stress or ablation results are substituted.',
        '','Workload is offered Gcycles per round in a coupled device-count/payload ramp. Arrival-rate CSV columns are empty; no tasks/s is inferred.',
        'CDF/CCDF curves use every eligible observation. Completion-latency curves exclude failed tasks; failure and deadline figures retain all attempts.',
        'Battery depletion/charge panels include battery-powered devices only (positive rated capacity); mains-powered devices are excluded from battery percentages.',
        'Device energy is task-attributed; infrastructure task energy excludes shared hover/baseline operation. No lifetime or hard real-time guarantee is inferred.',
        '','## Panel map','']
    for panel in ctx.records:notes.append(f"- {panel['id']}: {panel['title']} — {panel['status']}"+(f" — {panel['filename']}" if panel['filename'] else '')+(f". {panel['reason']}" if panel['reason'] else ''))
    (ctx.output/'README.md').write_text('\n'.join(notes),encoding='utf-8')
    (ctx.output/'run_status.json').write_text(json.dumps({'status':'complete','seeds':list(ctx.seeds),'generated_panels':sum(p['status']=='generated' for p in ctx.records),'planned_panels':len(ctx.records)}),encoding='utf-8')
    print('Generated',sum(p['status']=='generated' for p in ctx.records),'of',len(ctx.records),'panel slots.',flush=True)

def cli(direction,builder):
    parser=argparse.ArgumentParser(description=f'Agreed {direction} paper figure plan; completed evaluation data only.')
    parser.add_argument('--phase',type=Path,default=DEFAULT_PHASE)
    parser.add_argument('--seeds',default='1,2,3',help='Comma-separated completed seed indices')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'visualization'/direction)
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--training-only',action='store_true',help='Generate the four linked training panels only')
    parser.add_argument('--skip-training',action='store_true',help='Generate evaluation panels only')
    parser.add_argument('--training-seed-index',type=int,help='Required if the linked training run contains multiple training seeds')
    args=parser.parse_args();seeds=tuple(int(x) for x in args.seeds.split(','))
    if not seeds or len(set(seeds))!=len(seeds) or min(seeds)<1:parser.error('Seed indices must be positive and unique')
    if args.training_only and args.skip_training:parser.error('Choose training-only or skip-training, not both')
    from visualization.supportive.training_panels import render_training
    if args.training_only:
        render_training(args.phase,seeds,args.output_dir,direction,args.training_seed_index,args.validate_only)
        return
    ctx=Context(args.phase,seeds,args.output_dir)
    print(f'Loaded {len(ctx.data):,} attempts from {len(ctx.sources)} completed model/seed sources.',flush=True)
    if args.validate_only:return
    render(ctx,builder(),direction)
    if not args.skip_training:
        render_training(args.phase,seeds,args.output_dir,direction,args.training_seed_index)
