import wandb
from omegaconf import OmegaConf

def init(config):
    wandb.init(project = config.wandb.project,
               name = config.wandb.name, 
               notes = config.wandb.notes,
               tags = config.wandb.tags,
               group = config.wandb.group,
               resume = config.wandb.resume, 
               config = OmegaConf.to_container(config))

def log(log_dict):
    if 'step' in log_dict:
        step_info = log_dict.pop('step')
        wandb.log(log_dict, step = step_info)
    elif 'epoch' in log_dict:
        step_info = log_dict.pop('epoch')
        wandb.log(log_dict, step = step_info)
    else:
        wandb.log(log_dict)
        

    
