"""Delay and service reliability paper: six agreed four-panel groups.

Run: python visualization/delay_plotter.py
Training only: add --training-only. Evaluation only: add --skip-training.
Four linked training panels are also generated in the output training/ subfolder.
Default output: visualization/delay. New stress/ablation data are never fabricated.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from visualization.paper_plotting import (Panel,MODELS,CLASSES,LABELS,COLORS,cli,
    matrix,distribution,workload_curve,line_range)

def outcomes(ctx,ax,code):
    rows=[];components=['on_time','late','failed'];colors=['#3A9D70','#E6A23C','#777777']
    bottom=np.zeros(6)
    for metric,color in zip(components,colors):
        values=[]
        for model in MODELS:
            vals=[]
            for seed in ctx.seeds:
                g=ctx.subset(model);g=g[g.seed.eq(seed)];v=100*g[metric].mean();vals.append(v)
                rows.append({'model':model,'seed':seed,'outcome':metric,'percentage':v,'count':int(g[metric].sum()),'attempts':len(g)})
            values.append(np.mean(vals))
        ax.bar(range(6),values,bottom=bottom,color=color,width=.7);bottom+=values
    ax.set_xticks(range(6),[LABELS[m] for m in MODELS],rotation=32,ha='right')
    ax.set_ylabel('All attempted tasks (%)');ax.set_ylim(0,105)
    ax.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,['On time','Late','Failed'])],loc='lower right',fontsize=6)
    ctx.add_table(code,pd.DataFrame(rows))

def failure_reasons(ctx,ax,code):
    d=ctx.data;counts=d.loc[d.failed,'network_failure_reason'].fillna('Unspecified').replace('','Unspecified').value_counts()
    reasons=counts.index.tolist()
    if not reasons:
        ax.set_axis_off();ax.text(.5,.5,'No failed attempts in selected seeds',ha='center',transform=ax.transAxes)
        ctx.add_table(code,pd.DataFrame([{'failed_attempts':0}]));return
    rows=[];values=np.zeros((6,len(reasons)))
    for i,model in enumerate(MODELS):
        for j,reason in enumerate(reasons):
            vals=[]
            for seed in ctx.seeds:
                g=ctx.subset(model);g=g[g.seed.eq(seed)]
                n=int((g.failed & g.network_failure_reason.fillna('Unspecified').replace('','Unspecified').eq(reason)).sum())
                rate=1000*n/len(g);vals.append(rate);rows.append({'model':model,'seed':seed,'reason':reason,'count':n,'per_1000_attempts':rate})
            values[i,j]=np.mean(vals)
    im=ax.imshow(values,aspect='auto',cmap='Blues');ax.grid(False)
    ax.set_xticks(range(len(reasons)),[str(r).replace('_','\n') for r in reasons],fontsize=5.5,rotation=25,ha='right')
    ax.set_yticks(range(6),[LABELS[m] for m in MODELS]);ax.figure.colorbar(im,ax=ax,pad=.02,fraction=.045,label='Failures / 1,000 attempts')
    for i in range(6):
        for j in range(len(reasons)):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=6,
            color='white' if values[i,j]>(values.max()+values.min())/2 else 'black')
    ctx.add_table(code,pd.DataFrame(rows))

def miss_bursts(ctx,ax,code):
    rows=[];grid=np.arange(int(ctx.devices.longest_miss.max())+1)
    for model in MODELS:
        curves=[]
        for seed in ctx.seeds:
            v=np.sort(ctx.devices.loc[ctx.devices.model.eq(model)&ctx.devices.seed.eq(seed),'longest_miss'].to_numpy())
            y=np.searchsorted(v,grid,side='right')/len(v);curves.append(y)
            rows += [{'model':model,'seed':seed,'longest_streak':x,'cdf':q} for x,q in zip(grid,y)]
        line_range(ax,grid,curves,model)
    ax.set_xlabel('Longest unsuccessful streak per device (tasks)');ax.set_ylabel('Cumulative fraction of devices');ax.set_ylim(0,1.02)
    ctx.add_table(code,pd.DataFrame(rows))

def plan():
    groups=[]
    groups.append(('R1_Deadline_reliability',[
        *[Panel(f'R1{chr(97+i)}',f'{cls}: deadline success','line with seed range',
          lambda c,a,k,cls=cls:workload_curve(c,a,k,cls)) for i,cls in enumerate(CLASSES)],
        Panel('R1d','Lower-tail device success','heatmap',lambda c,a,k:matrix(c,a,k,
            lambda g:g.success_pct.quantile(.1),'Device-success P10 (%)',device=True))]))
    groups.append(('R2_Latency_distributions_and_tails',[
        *[Panel(f'R2{chr(97+i)}',f'{cls}: full latency tail','CCDF',
          lambda c,a,k,cls=cls:distribution(c,a,k,cls,'observed_complete_latency_ms',ccdf=True)) for i,cls in enumerate(CLASSES)],
        Panel('R2d','Completion latency P99','heatmap',lambda c,a,k:matrix(c,a,k,
            lambda g:g.loc[g.complete,'observed_complete_latency_ms'].quantile(.99),'Mean seed P99 (ms)',fmt='.1f'))]))
    parts=[('radio_ms','Transmission and propagation'),('queue_delay_ms','Queueing'),
           ('processing_delay_ms','Processing'),('handover_delay_component_ms','Handover')]
    groups.append(('R3_Sources_of_delay',[
        Panel('R3'+chr(97+i),title,'heatmap',lambda c,a,k,col=col:matrix(c,a,k,
            lambda g:g.loc[g.complete,col].mean(),'Mean completed-task delay (ms)',fmt='.3f')) for i,(col,title) in enumerate(parts)]))
    groups.append(('R4_Unsuccessful_service',[
        Panel('R4a','On-time, late and failed tasks','stacked bars',outcomes),
        Panel('R4b','Recorded failure causes','heatmap',failure_reasons),
        Panel('R4c','Late completion under workload','line with seed range',lambda c,a,k:workload_curve(c,a,k,
            metric='late',ylabel='Late but completed attempts (%)')),
        Panel('R4d','Unsuccessful-update bursts','ECDF of longest per-device streak',miss_bursts)]))
    groups.append(('R5_Controlled_robustness',[
        Panel('R5a','Application deadline tightness','controlled line',pending='Requires independently varied deadlines and re-evaluation; not a counterfactual relabelling of logged outcomes.'),
        Panel('R5b','Alarm burst intensity','controlled line',pending='Requires controlled burst-traffic experiments at fixed device count and background load.'),
        Panel('R5c','Channel impairment','controlled line',pending='Requires matched channel-impairment experiments.'),
        Panel('R5d','Mobility and handover exposure','controlled line',pending='Requires independently controlled mobility experiments.')]))
    groups.append(('R6_Mechanism_contribution',[
        Panel('R6a','Deadline mechanism ablation','grouped bars',pending='Requires a specified new mechanism, retrained ablation policies and matched evaluation.'),
        Panel('R6b','Mobility mechanism ablation','grouped bars',pending='Requires a mobility mechanism and its controlled ablation; current model differences are not ablations.'),
        Panel('R6c','Reliability under energy budgets','controlled line',pending='Requires matched energy-budget operating points.'),
        Panel('R6d','Decision overhead and budget','ECDF with budget reference',pending='Inference time is not recorded; measure decision latency and define the deployment platform.')]))
    return groups

if __name__=='__main__':cli('delay',plan)
