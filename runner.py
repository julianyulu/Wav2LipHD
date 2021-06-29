import os
import argparse
import importlib 
from omegaconf import OmegaConf 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--config",
                        type = str,
                        required = True,
                        help = "yaml config file")
    parser.add_argument("-g",
                        "--gpuid",
                        type = str,
                        required = True,
                        help = "gpu id (one or more, seperated by ,")
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    
    run_config = OmegaConf.load(args.config)
    base_config = OmegaConf.load(run_config.base_config)
    config = OmegaConf.merge(run_config, base_config)

    print("================ CONFIG ===============")
    print(OmegaConf.to_yaml(config))

    task = config.task 
    if task == 'Wav2LipHD':
        module_path = "Wav2LipHD"
    elif task == 'SyncNetHD':
        module_path = "SyncNetHD"
    elif task == 'SyncNet':
        module_path = "SyncNet"
    elif task == 'LdmkSync':
        module_path = "LdmkSync"
    elif task == 'Wav2LipHD_patchGAN':
        module_path = "Wav2LipHD"
    else:
        raise ValueError(f"{task} name not valid, options: ['Wav2LipHD', SyncNetHD']")

    if 'patch' in task:
        Runner = getattr(importlib.import_module(f"{module_path}.runner_patch"),'Runner')
    else:
        Runner = getattr(importlib.import_module(f"{module_path}.runner"),'Runner')
    runner = Runner(config)
    runner.train()
