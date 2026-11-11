"""Checks that training figures cannot silently consume evaluation observations."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from visualization.paper_plotting import MODELS
from visualization.supportive.training_panels import resolve_training,summarise_rounds

class TrainingPanelTests(unittest.TestCase):
    def fixture(self,root,training_seeds=(1,)):
        root=Path(root);phase=root/'evaluation';phase.mkdir();run=root/'training';exploration=run/'exploration';exploration.mkdir(parents=True)
        (phase/'manifest.json').write_text(json.dumps({'mode':'eval','seeds':{'seed_2':123}}))
        (exploration/'manifest.json').write_text(json.dumps({'mode':'train','seeds':{f'seed_{s}':456+s for s in training_seeds}}))
        for m in MODELS:
            folder=phase/'seed_2/offloading'/m;folder.mkdir(parents=True)
            (folder/'job_completion.json').write_text(json.dumps({'model':m,'training':False,'seed':123,'training_run':str(run),'configuration':{'rounds':200}}))
            for seed in training_seeds:
                folder=exploration/f'seed_{seed}/offloading'/m;folder.mkdir(parents=True)
                (folder/'job_completion.json').write_text(json.dumps({'configuration':{'rounds':200}}))
        return phase,run
    def test_evaluation_seed_does_not_select_training_seed(self):
        with tempfile.TemporaryDirectory() as f:
            phase,run=self.fixture(f)
            home,_,_=resolve_training(phase,[2])
            self.assertEqual(home,run/'exploration/seed_1')
    def test_ambiguous_training_seed_requires_selection(self):
        with tempfile.TemporaryDirectory() as f:
            phase,run=self.fixture(f,(1,3))
            with self.assertRaisesRegex(ValueError,'Multiple training seeds'):resolve_training(phase,[2])
            self.assertEqual(resolve_training(phase,[2],3)[0],run/'exploration/seed_3')
    def test_missing_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as f:
            phase,_=self.fixture(f);path=phase/'seed_2/offloading/dqn/job_completion.json';d=json.loads(path.read_text());d['training_run']=None;path.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'provenance'):resolve_training(phase,[2])
    def test_configuration_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as f:
            phase,run=self.fixture(f);path=run/'exploration/seed_1/offloading/dqn/job_completion.json';path.write_text(json.dumps({'configuration':{'rounds':201}}))
            with self.assertRaisesRegex(ValueError,'configuration mismatch'):resolve_training(phase,[2])
    def test_failed_tasks_remain_in_energy_reward_and_success_denominators(self):
        d=pd.DataFrame({'round':[1,1,1],'task_id':['a','b','c'],'reward':[3,2,-5],
            'action_entropy':[.1,.2,.3],'success':[1,0,0],'status':['completed','completed','failed'],
            'observed_complete_latency_ms':[1,5,np.nan],'deadline_ms':[2,2,2],
            'device_battery_energy_j':[.001,.002,.006]})
        row=summarise_rounds(d).iloc[0]
        self.assertEqual(row.attempts,3);self.assertAlmostEqual(row.mean_reward,0)
        self.assertAlmostEqual(row.deadline_success_pct,100/3)
        self.assertAlmostEqual(row.device_energy_mj_per_attempt,3)
        self.assertAlmostEqual(row.completed_latency_ms,3)

if __name__=='__main__':unittest.main()
