"""Regression checks for scientific accounting and completed-only plot sources."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from visualization.paper_plotting import COLS,TEXT,MODELS,Context,energy_per_useful,longest_miss_streak,load_evaluation
from visualization.delay_plotter import plan as delay_plan
from visualization.energy_plotter import plan as energy_plan

class PaperPlotterTests(unittest.TestCase):
    def test_energy_denominator_retains_unsuccessful_energy(self):
        self.assertEqual(energy_per_useful([2,3,5],[True,False,False]),10)
    def test_zero_useful_tasks_is_undefined(self):
        self.assertTrue(np.isnan(energy_per_useful([2,3],[False,False])))
    def test_burst_resets_on_success_and_missing_round(self):
        self.assertEqual(longest_miss_streak([False,False,True,False],[1,2,3,4]),2)
        self.assertEqual(longest_miss_streak([False,False],[1,3]),1)
        self.assertEqual(longest_miss_streak([True,True],[1,2]),0)
    def test_agreed_panel_plans(self):
        for plan in [delay_plan,energy_plan]:
            groups=plan();panels=[p for _,ps in groups for p in ps]
            self.assertEqual(len(groups),6);self.assertEqual(len(panels),24)
            self.assertEqual(len({p.code for p in panels}),24)
            self.assertEqual(sum(p.draw is not None for p in panels),16)
            self.assertTrue(all(p.draw is not None or bool(p.pending) for p in panels))
    def fixture(self,folder):
        root=Path(folder);(root/'manifest.json').write_text(json.dumps({'mode':'eval','seeds':{'seed_7':123}}))
        for model in MODELS:
            home=root/'seed_7/offloading'/model;home.mkdir(parents=True)
            rows=[]
            for i,cls in enumerate(['URLLC','mMTC','eMBB']):
                row={c:1.0 for c in COLS if c not in TEXT}
                row.update({c:'test' for c in TEXT})
                row.update(task_id=str(i),class_name=cls,status='completed',success=1,round=1 if i==0 else 2,
                    device_id=i,deadline_ms=100,observed_complete_latency_ms=1)
                rows.append(row)
            pd.DataFrame(rows).to_csv(home/'results.csv',sep=';',index=False)
            (home/'job_completion.json').write_text(json.dumps({'training':False,'seed':123,'decisions':3,'rounds':2}))
        return root
    def test_explicit_seed_index_and_offered_workload(self):
        with tempfile.TemporaryDirectory() as f:
            root=self.fixture(f);data,sources=load_evaluation(root,[7])
            self.assertEqual(set(data.seed),{7});self.assertEqual(len(data),18);self.assertEqual(len(sources),6)
            self.assertTrue(np.allclose(data.loc[data['round'].eq(2),'workload'],2e-9))
    def test_mains_powered_device_excluded_from_battery_distribution(self):
        with tempfile.TemporaryDirectory() as f:
            root=self.fixture(f);p=root/'seed_7/offloading/dqn/results.csv'
            data=pd.read_csv(p,sep=';');data.loc[1,['battery_capacity_j','device_battery_before_j','device_battery_after_j']]=0;data.to_csv(p,sep=';',index=False)
            ctx=Context(root,[7],root/'figures')
            self.assertEqual(len(ctx.data),18)
            self.assertEqual(len(ctx.devices),17)
    def test_partial_source_rejected(self):
        with tempfile.TemporaryDirectory() as f:
            root=self.fixture(f);p=root/'seed_7/offloading/dqn/job_completion.json';meta=json.loads(p.read_text());meta['decisions']=4;p.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'Incomplete'):load_evaluation(root,[7])
    def test_wrong_seed_rejected(self):
        with tempfile.TemporaryDirectory() as f:
            root=self.fixture(f);p=root/'seed_7/offloading/dqn/job_completion.json';meta=json.loads(p.read_text());meta['seed']=456;p.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'Wrong phase or seed'):load_evaluation(root,[7])
    def test_inconsistent_success_rejected(self):
        with tempfile.TemporaryDirectory() as f:
            root=self.fixture(f);p=root/'seed_7/offloading/dqn/results.csv';data=pd.read_csv(p,sep=';');data.loc[0,'success']=0;data.to_csv(p,sep=';',index=False)
            with self.assertRaisesRegex(ValueError,'Success flag disagrees'):load_evaluation(root,[7])
    def test_individual_png_export_contract(self):
        from unittest.mock import patch
        from visualization.paper_plotting import Panel,render,save_panel,panel_filename
        self.assertEqual(panel_filename('URLLC: deadline success'),'URLLC - deadline success.png')
        with tempfile.TemporaryDirectory() as f:
            root=self.fixture(f);ctx=Context(root,[7],Path(f)/'plots')
            def check_and_save(fig,target):
                self.assertEqual(len(fig.axes),1)
                self.assertEqual(fig.axes[0].get_title(),'')
                self.assertEqual(fig._suptitle.get_text(),'')
                self.assertIsNotNone(fig.axes[0].get_legend())
                save_panel(fig,target)
            panels=[Panel('T1','URLLC: deadline success','line',lambda c,a,k:a.plot([0,1],[0,1])),
                    Panel('T2','Pending','line',pending='Needs data')]
            with patch('visualization.paper_plotting.save_panel',side_effect=check_and_save):
                render(ctx,[('Test',panels)],'delay')
            self.assertEqual([x.name for x in ctx.output.glob('*.png')],['URLLC - deadline success.png'])
            self.assertFalse(list(ctx.output.glob('*.pdf')))
            self.assertEqual(ctx.records[1]['status'],'pending')
if __name__=='__main__':unittest.main()
