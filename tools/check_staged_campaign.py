"""Small end-to-end checkpoint and evaluation check; no legacy checkpoint writes."""
import sys, tempfile, json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import config as cfg
from applications import MainApplication as app
from applications.staged_campaign import prepare_job
from scheduler.common.comparison_runner import create_comparison_scheduler
from scheduler.common import common_config
from config.topology import get_deployment_profile

def main():
    cfg.ROUNDS=6;cfg.MAX_DEVICES=32;cfg.WORKLOAD_OVERRIDES={'start_devices':32,'max_devices':32,'rounds':6}
    profile=get_deployment_profile(cfg.SIMULATION['deployment_profile'])
    assert profile.ground_bs_count==81 and profile.uav_profile_id=='nr-fr1-air'
    assert cfg.MULTISEED_COUNT==cfg.TRAINING_SEED_COUNT==1
    with tempfile.TemporaryDirectory(prefix='aiot_staged_smoke_') as folder:
        home=Path(folder);train=home/'train'; results=[]
        for model in cfg.RUN_MODELS:
            spec={'train':True,'model_name':model,'seed':1234,'seed_index':0,'seed_total':1,'output_dir':str(train/'exploration/seed_1'),'mirror_legacy':False}
            app._run_one_job(spec)
            paths=list((train/'exploration/seed_1/offloading'/model).glob('*policy.pt'))+list((train/'exploration/seed_1/mobility'/model).glob('*.pth'))
            before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            spec.update(train=False,seed=5678,training_run=str(train),output_dir=str(home/'eval'/'evaluation/seed_1'))
            app._run_one_job(spec)
            assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            diag=json.loads((home/'eval/evaluation/seed_1/offloading'/model/f'diagnostics_{model}.json').read_text())
            assert diag['train_mode'] is False
            results.append({'model':model,'train_eval_passed':True,'evaluation_diagnostics':diag})
        from applications.staged_campaign import review_evaluation
        review_evaluation(home/'eval')
        from visualization.supportive.automatic_plotting import CampaignPlotCoordinator
        plots=CampaignPlotCoordinator(train/'exploration',cfg.RUN_MODELS,1)
        for model in cfg.RUN_MODELS:
            plots.record_job_completion({'model':model,'seed_index':0})
        plots.finalize()
        assert (train/'exploration/plots/plot_manifest.json').is_file()
        try:
            create_comparison_scheduler('dqn',config=common_config,output_dir=home/'missing',checkpoint_root=home/'missing_checkpoint',strict_load=True)
        except FileNotFoundError:pass
        else:raise AssertionError('Missing checkpoint was accepted')
        manifest=train/'exploration/seed_1/offloading/dqn/checkpoint_manifest.json'
        data=json.loads(manifest.read_text());data['configuration']['deployment']='incorrect';manifest.write_text(json.dumps(data))
        try:prepare_job({'train':False,'model_name':'dqn','training_run':str(train)})
        except ValueError:pass
        else:raise AssertionError('Mismatched deployment was accepted')
        destination=ROOT/'.backups'/'staged_smoke_validation.json';destination.write_text(json.dumps(results,indent=2))
        print('SMOKE PASSED: all six models, 192 train + 192 evaluation decisions each; strict missing/mismatched checkpoint rejection; checkpoint files unchanged during evaluation.',flush=True)
if __name__=='__main__':main()
