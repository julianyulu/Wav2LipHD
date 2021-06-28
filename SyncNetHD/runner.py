import os
import glob 
import torch
import argparse
from tqdm import tqdm 
from .dataset import Dataset
from .model import SyncNetColor,SyncNetColorHD
from omegaconf import OmegaConf 
from torch.utils import data as data_utils
from utils import wandb_utils
from utils.batch_sampler import RandomEntryBatchSampler
import pdb 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Runner:
    def __init__(self, config):
        self.cfg = config 
        # set model 
        if self.cfg.model.use_SyncNetHD: 
            self.model = SyncNetColorHD()
        else:
            self.model = SyncNetColor()
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr = self.cfg.runtime.init_learning_rate)
        
        if self.cfg.model.resume_ckpt:
            self.resume_from_ckpt(self.cfg.model.resume_ckpt)
        else:
            os.makedirs(self.cfg.runtime.checkpoint_dir, exist_ok = True)
            OmegaConf.save(self.cfg, os.path.join(self.cfg.runtime.checkpoint_dir, 'config.yaml'))

        if self.cfg.wandb.enable:
            wandb_utils.init(self.cfg)

    def resume_from_ckpt(self, resume_ckpt):
        # overwrite curr config from ckpt dir 
        if self.cfg.model.resume_ckpt_config: 
            self.cfg = OmegaConf.load(os.path.join(os.path.dirname(resume_ckpt), 'config.yaml'))
            
        print("Loading checkpoint from: {}".format(resume_ckpt))

        if torch.cuda.is_available():
            checkpoint = torch.load(resume_ckpt, map_location = lambda storage, loc: storage)
        else:
            checkpoint = torch.load(resume_ckpt)

        self.model.load_state_dict(checkpoint["state_dict"])

        if not self.cfg.runtime.reset_optimizer:
            optimizer_state = checkpoint["optimizer"]
            if optimizer_state is not None:
                print("Loading optimizer state from {}".format(resume_ckpt))
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                
        self.global_step = checkpoint["global_step"]
        self.global_epoch = checkpoint["global_epoch"]

    def save_checkpoint(self):
        os.makedirs(self.cfg.runtime.checkpoint_dir, exist_ok = True)
        ckpt_path = os.path.join(self.cfg.runtime.checkpoint_dir,
                                f"checkpoint_step{int(self.global_step)}.pth")
        opt_state = self.optimizer.state_dict() if self.cfg.runtime.save_optimizer_state else None
        model_state = self.model.state_dict()

        torch.save({
            "state_dict": model_state,
            "optimizer": opt_state,
            "global_step": self.global_step,
            "global_epoch": self.global_epoch
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
                   
    def get_dataloader(self, split):
        dataset = Dataset(self.cfg, split)
        if split == 'train':
            batch_sampler = RandomEntryBatchSampler(len(dataset),
                                                    batch_size = self.cfg.runtime.batch_size,
                                                    steps_per_epoch = self.cfg.runtime.steps_per_epoch)
            data_loader = data_utils.DataLoader(dataset,
                                                pin_memory = True,
                                                batch_sampler = batch_sampler)
        else:
            
            data_loader = data_utils.DataLoader(dataset,
                                                batch_size = self.cfg.runtime.batch_size,
                                                pin_memory = False,
                                                num_workers = self.cfg.runtime.num_workers)
        return data_loader
    
    def loss_fn(self, a, v, y):
        if not hasattr(self, '_bce_loss'):
            self._bce_loss = torch.nn.BCELoss()
        d = torch.nn.functional.cosine_similarity(a, v)
        loss = self._bce_loss(d.unsqueeze(1), y)
        return loss 

    def log(self, log_dict):
        if self.cfg.wandb.enable:
            wandb_utils.log(log_dict)
                
    def eval(self):
        if not hasattr(self, '_test_data_loader'):
            self._test_data_loader = self.get_dataloader('val')
        losses = []
        step = 0 
        while True:
            for x, mel, y in self._test_data_loader:
                self.model.eval()
                x = x.to(device)
                mel = mel.to(device)
                a, v = self.model(mel, x)
                y = y.to(device)
                loss = self.loss_fn(a, v, y)
                losses.append(loss.item())
                step += 1
                if step >= self.cfg.runtime.eval_forward_steps:
                    break
            if step >= self.cfg.runtime.eval_forward_steps:
                    break
                
        averaged_loss = sum(losses) / len(losses)
        
        return {"eval_loss": averaged_loss,
                "step": self.global_step,
                "epoch": self.global_epoch}

    def train(self):
        if not hasattr(self, 'global_step'): self.global_step = 0
        if not hasattr(self, 'global_epoch'): self.global_epoch = 0

        train_data_loader = self.get_dataloader('train')

        while self.global_epoch < self.cfg.runtime.nepochs:
            running_loss = 0.
            prog_bar = tqdm(enumerate(train_data_loader), total = self.cfg.runtime.steps_per_epoch)
        
            for step, (x, mel, y) in prog_bar:
                self.model.train()
                self.optimizer.zero_grad()
                x = x.to(device)
                mel = mel.to(device)
                a, v = self.model(mel, x)
                y = y.to(device)
                loss = self.loss_fn(a, v, y)
                loss.backward()
                self.optimizer.step()
                self.global_step += 1
                running_loss += loss.item()

                prog_bar.set_description(f"Epoch: {self.global_epoch} | Step: {self.global_step} | Train Loss: {running_loss / (step + 1):.6f}")
                
                if self.global_step >0 and self.global_step % self.cfg.runtime.checkpoint_interval == 0:
                    self.save_checkpoint()

                if self.global_step % self.cfg.runtime.eval_interval == 0:
                    with torch.no_grad():
                        eval_res = self.eval()
                    self.log(eval_res)
                    print(f"\nEval Loss @ step {self.global_step} | epoch {self.global_epoch}: {eval_res['eval_loss']:.6f}")        
                    
            self.global_epoch += 1 
            self.log({"step": self.global_step,
                      "epoch": self.global_epoch,
                      "train_loss": running_loss / (step + 1)})
