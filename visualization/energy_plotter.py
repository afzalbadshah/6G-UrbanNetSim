"""Device battery preservation paper: six agreed four-panel groups.

Run: python visualization/energy_plotter.py
Training only: add --training-only. Evaluation only: add --skip-training.
Four linked training panels are also generated in the output training/ subfolder.
Default output: visualization/energy. Battery lifetime and ablations require new data.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from visualization.paper_plotting import (Panel,MODELS,CLASSES,LABELS,COLORS,cli,
    matrix,distribution,workload_curve,line_range,model_bars,energy_per_useful)

def battery_tail(ctx,ax,code):
    rows=[];markers=['o','^','x'];names=['Median','P90','Worst']
    for i,model in enumerate(MODELS):
        for j,(name,q) in enumerate(zip(names,[.5,.9,1.])):
            vals=[]
            for seed in ctx.seeds:
                g=ctx.devices[ctx.devices.model.eq(model)&ctx.devices.seed.eq(seed)]
                v=g.depletion.quantile(q);vals.append(v);rows.append({'model':model,'seed':seed,'statistic':name,'depletion_pp':v,'devices':len(g)})
            ax.scatter(np.mean(vals),i+(j-1)*.15,marker=markers[j],s=25,color=COLORS[model])
    ax.set_yticks(range(6),[LABELS[m] for m in MODELS]);ax.set_xlabel('Battery depletion (percentage points)');ax.set_xlim(left=0)
    ax.legend(handles=[Line2D([],[],linestyle='none',marker=m,color='#555555',label=n) for m,n in zip(markers,names)],loc='lower right',fontsize=6)
    ctx.add_table(code,pd.DataFrame(rows))

def energy_stack(ctx,ax,code):
    parts=[('radio_energy_j','Tx + Rx','#4C78A8'),('device_local_compute_energy_j','Local CPU','#F2A541'),
           ('device_wait_energy_j','Waiting','#59A14F')]
    rows=[];bottom=np.zeros(6)
    for col,label,color in parts:
        vals=[]
        for model in MODELS:
            seed_vals=[]
            for seed in ctx.seeds:
                g=ctx.subset(model);g=g[g.seed.eq(seed)];v=1000*g[col].mean();seed_vals.append(v)
                rows.append({'model':model,'seed':seed,'component':label,'mean_mj_per_attempt':v})
            vals.append(np.mean(seed_vals))
        ax.bar(range(6),vals,bottom=bottom,color=color,width=.7);bottom+=vals
    # Preserve any additional recorded device task energy in the decomposition.
    residual=[]
    for model in MODELS:
        seed_vals=[]
        for seed in ctx.seeds:
            g=ctx.subset(model);g=g[g.seed.eq(seed)]
            difference=g.device_battery_energy_j-g[[c for c,_,_ in parts]].sum(axis=1)
            if (difference < -1e-8).any():raise ValueError('Device components exceed battery task energy')
            v=1000*difference.clip(lower=0).mean();seed_vals.append(v)
            rows.append({'model':model,'seed':seed,'component':'Other device','mean_mj_per_attempt':v})
        residual.append(np.mean(seed_vals))
    handles=[Patch(color=c,label=l) for _,l,c in parts]
    if max(residual)>1e-6:
        ax.bar(range(6),residual,bottom=bottom,color='#999999',width=.7);handles.append(Patch(color='#999999',label='Other device'))
    ax.set_xticks(range(6),[LABELS[m] for m in MODELS],rotation=32,ha='right');ax.set_ylabel('Mean device energy (mJ/attempt)');ax.set_ylim(bottom=0)
    ax.legend(handles=handles,loc='upper right',fontsize=5.8);ctx.add_table(code,pd.DataFrame(rows))

def useful_tradeoff(ctx,ax,code):
    rows=[];markers=['o','s','^']
    for model in MODELS:
        for cls,marker in zip(CLASSES,markers):
            xs=[];ys=[]
            for seed in ctx.seeds:
                g=ctx.subset(model,cls);g=g[g.seed.eq(seed)]
                x=1000*energy_per_useful(g.device_battery_energy_j,g.on_time);y=100*g.on_time.mean();xs.append(x);ys.append(y)
                rows.append({'model':model,'seed':seed,'class_name':cls,'device_mj_per_useful_task':x,'deadline_success_pct':y})
            ax.scatter(np.nanmean(xs),np.nanmean(ys),marker=marker,s=26,color=COLORS[model],edgecolors='white',linewidth=.3)
    ax.set_xscale('log');ax.set_xlabel('Device energy per useful task (mJ)');ax.set_ylabel('Deadline success (%)')
    ax.legend(handles=[Line2D([],[],linestyle='none',marker=m,color='#555555',label=c) for c,m in zip(CLASSES,markers)],loc='lower left',fontsize=6)
    ctx.add_table(code,pd.DataFrame(rows))

def fixed_cohort(ctx,ax,code,charge=False):
    rows=[];rounds=np.arange(50,201)
    for model in MODELS:
        curves=[]
        for seed in ctx.seeds:
            d=ctx.subset(model,'mMTC' if charge else None);d=d[d.seed.eq(seed)]
            if charge:d=d[d.battery_capacity_j.gt(0)]
            # Same IDs/activation window within a seed; late entrants never dilute the curve.
            early=set(d.loc[d['round'].eq(50),'device_id']);last=set(d.loc[d['round'].eq(200),'device_id']);ids=early&last
            g=d[d.device_id.isin(ids)&d['round'].between(50,200)]
            if not ids:raise ValueError('No complete fixed cohort')
            if charge:
                if (g.battery_capacity_j<=0).any():raise ValueError('Non-positive capacity')
                series=(100*g.device_battery_after_j/g.battery_capacity_j).groupby(g['round']).mean().reindex(rounds)
            else:series=g.groupby('round').device_battery_energy_j.sum().reindex(rounds).cumsum()
            curves.append(series.to_numpy())
            rows += [{'model':model,'seed':seed,'round':int(r),'value':v,'cohort_devices':len(ids)} for r,v in zip(rounds,series)]
        line_range(ax,rounds,curves,model)
    ax.set_xlabel('Evaluation round (fixed cohort from round 50)')
    ax.set_ylabel('Battery-powered mMTC charge (%)' if charge else 'Cumulative device task energy (J)')
    ax.set_ylim((0,100) if charge else (0,None));ctx.add_table(code,pd.DataFrame(rows))

def energy_transfer(ctx,ax,code):
    rows=[]
    for j,(col,label,color) in enumerate([('device_battery_energy_j','Device','#4C78A8'),('infra_energy_j','Infrastructure','#999999')]):
        means=[]
        for model in MODELS:
            vals=[]
            for seed in ctx.seeds:
                g=ctx.subset(model);g=g[g.seed.eq(seed)];v=1000*g[col].mean();vals.append(v)
                rows.append({'model':model,'seed':seed,'accounting':label,'mean_mj_per_attempt':v})
            means.append(np.mean(vals))
        ax.bar(np.arange(6)+(j-.5)*.34,means,width=.34,color=color,label=label)
    ax.set_xticks(range(6),[LABELS[m] for m in MODELS],rotation=32,ha='right');ax.set_ylabel('Attributed task energy (mJ/attempt)');ax.set_ylim(bottom=0)
    ax.legend(fontsize=6);ctx.add_table(code,pd.DataFrame(rows))

def plan():
    groups=[]
    groups.append(('E1_Device_battery_depletion',[
        *[Panel('E1'+chr(97+i),cls+': battery depletion','ECDF',lambda c,a,k,cls=cls:distribution(c,a,k,cls,'depletion',device=True)) for i,cls in enumerate(CLASSES)],
        Panel('E1d','Median, P90 and worst depletion','percentile point comparison',battery_tail)]))
    groups.append(('E2_Device_energy_components',[
        *[Panel('E2'+chr(97+i),title,'heatmap',lambda c,a,k,col=col:matrix(c,a,k,
           lambda g:g[col].mean(),'Mean device energy (mJ/attempt)',scale=1000,fmt='.3f'))
          for i,(col,title) in enumerate([('radio_energy_j','Transmission and reception'),('device_local_compute_energy_j','Local computation'),('device_wait_energy_j','Waiting')])],
        Panel('E2d','All workloads: device energy','stacked bars',energy_stack)]))
    groups.append(('E3_Energy_for_useful_service',[
        *[Panel('E3'+chr(97+i),cls+': energy per useful task','bars with seed dots and ranges',
          lambda c,a,k,cls=cls:model_bars(c,a,k,lambda g:1000*energy_per_useful(g.device_battery_energy_j,g.on_time),
             'Device energy per useful task (mJ)',cls)) for i,cls in enumerate(CLASSES)],
        Panel('E3d','Energy and deadline success','scatter by model and category',useful_tradeoff)]))
    groups.append(('E4_Vulnerable_devices',[
        Panel('E4a','Initial-charge sensitivity','controlled line',pending='Requires controlled initial-charge groups; current charge differences are not an independent battery sweep.'),
        Panel('E4b','Low-battery deadline success','grouped bars',pending='Requires predefined low-charge cohorts under matched controlled exposure.'),
        Panel('E4c','Reserve-threshold crossings','controlled line',pending='Requires an agreed reserve threshold and independent workload/initial-charge experiments.'),
        Panel('E4d','Weak versus favorable channels','box plots',pending='Requires matched channel groups and initial charge; selected-route observations alone do not isolate channel effects.')]))
    groups.append(('E5_Battery_over_time_and_workload',[
        Panel('E5a','Fixed-cohort cumulative energy','line with seed range',fixed_cohort),
        Panel('E5b','Fixed mMTC cohort charge','line with seed range',lambda c,a,k:fixed_cohort(c,a,k,charge=True)),
        Panel('E5c','Energy under the coupled load ramp','line with seed range',lambda c,a,k:workload_curve(c,a,k,
            metric='device_battery_energy_j',scale=1000,ylabel='Mean device task energy (mJ)')),
        Panel('E5d','Time to battery reserve','survival curve with censoring',pending='Requires a predefined reserve threshold and longer fixed-cohort operation; no lifetime estimate is substituted.')]))
    groups.append(('E6_Mechanism_and_energy_transfer',[
        Panel('E6a','Battery mechanism ablation','grouped bars',pending='Requires a specified battery mechanism and retrained ablation policies.'),
        Panel('E6b','Energy at common reliability','controlled operating points',pending='Requires matched deadline-success operating points, not arbitrary comparisons at different service levels.'),
        Panel('E6c','All workloads: energy transfer','aligned grouped bars',energy_transfer),
        Panel('E6d','Placement-rule baselines','grouped bars',pending='Requires documented local-only, feasible-edge and cloud-placement evaluations.')]))
    return groups

if __name__=='__main__':cli('energy',plan)
