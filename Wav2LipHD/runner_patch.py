import os
import glob 
import torch
import wandb 
import argparse
import numpy as np 
from tqdm import tqdm
from torch.nn import functional as F
from .dataset import Dataset
from .model import Wav2LipHD, Wav2LipHD_disc, Wav2LipHD_disc_patch
from SyncNetHD.model import SyncNetColor, SyncNetColorHD
from omegaconf import OmegaConf 
from torch.utils import data as data_utils
from utils import wandb_utils
from utils.batch_sampler import RandomEntryBatchSampler
import pdb 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Runner:
    def __init__(self, config):
        self.cfg = config 
        self.model = Wav2LipHD()
        # set SyncNet
        if self.cfg.model.use_SyncNetHD: 
            self.syncnet = SyncNetColorHD()
        else:
            self.syncnet = SyncNetColor()
        # set Multi-GPU
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.syncnet = torch.nn.DataParallel(self.syncnet)
        self.model = self.model.to(device)
        self.syncnet = self.syncnet.to(device)
        # set Discriminator (optional)
        if self.cfg.model.disc_weight > 0:
            self.disc = Wav2LipHD_disc()
            if torch.cuda.device_count() > 1:
                self.disc = torch.nn.DataParallel(self.disc)
            self.disc = self.disc.to(device)
            self.disc_optimizer = torch.optim.Adam([p for p in self.disc.parameters() if p.requires_grad], lr = self.cfg.runtime.disc_learning_rate, betas = (0.5, 0.999))
        else:
            self.disc = None
        # set patchGAN discriminator (optional)
        if self.cfg.model.patchgan_weight > 0:
            self.patchgan = Wav2LipHD_disc_patch()
            if torch.cuda.device_count() > 1:
                self.patchgan = torch.nn.DataParallel(self.patchgan)
            self.patchgan = self.patchgan.to(device)
            self.patchgan_optimizer = torch.optim.Adam([p for p in self.patchgan.parameters() if p.requires_grad], lr = self.cfg.runtime.patchgan_learning_rate, betas = (0.5, 0.999))
            
        # set optimizer 
        self.model_optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr = self.cfg.runtime.model_learning_rate, betas = (0.5, 0.999))
        # load pretrained ckpt (optional)
        self.init_model_params()
        # prepare output dir 
        os.makedirs(self.cfg.runtime.checkpoint_dir, exist_ok = True)
        OmegaConf.save(self.cfg, os.path.join(self.cfg.runtime.checkpoint_dir, 'config.yaml'))
        # init wandb 
        if self.cfg.wandb.enable:
            wandb_utils.init(self.cfg)

    def init_model_params(self):
        if self.cfg.model.syncnet_ckpt:
            self.load_checkpoint(self.cfg.model.syncnet_ckpt, self.syncnet, reset_optimizer = True)
            print(f"[*] Loaded SyncNet ckpt: {self.cfg.model.syncnet_ckpt}")
        if self.cfg.model.wav2lip_ckpt:
            step, epoch = self.load_checkpoint(self.cfg.model.wav2lip_ckpt, self.model, optimizer = self.model_optimizer)
            if os.path.abspath(os.path.dirname(self.cfg.model.wav2lip_ckpt)) == os.path.abspath(self.cfg.runtime.checkpoint_dir):
                self.global_step = step
                self.global_epoch = epoch
                print(f"[*] Setting global epoch: {epoch}, global step: {step}")
            print(f"[*] Loaded Wav2Lip ckpt: {self.cfg.model.wav2lip_ckpt}")
        if self.cfg.model.disc_ckpt:
            self.load_checkpoint(self.cfg.model.disc_ckpt, self.disc, optimizer = self.disc_optimizer)
            print(f"[*] Loaded Wav2Lip_disc ckpt: {self.cfg.model.disc_ckpt}")
        # freeze syncnet parameters
        for p in self.syncnet.parameters(): p.requires_grad = False

        
    @staticmethod
    def load_checkpoint(ckpt_path, model, optimizer = None, reset_optimizer=False):
        print("Loading checkpoint from: {}".format(ckpt_path))
        if torch.cuda.is_available():
            checkpoint = torch.load(ckpt_path, map_location = lambda storage, loc: storage)
        else:
            checkpoint = torch.load(ckpt_path)

        if 'module.' in list(checkpoint['state_dict'].keys())[0]:
            state_dict = checkpoint['state_dict']
            #state_dict = {k.replace('module.', ''): v for k,v in checkpoint["state_dict"].items()}
            model.load_state_dict(state_dict)
        else:
            state_dict = {'module.' + k: v for k,v in checkpoint["state_dict"].items()}
            model.load_state_dict(state_dict)
                    
        if not reset_optimizer:
            optimizer_state = checkpoint["optimizer"]
            if optimizer_state is not None:
                print("Loading optimizer state from {}".format(ckpt_path))
                optimizer.load_state_dict(checkpoint["optimizer"])

                
        return checkpoint["global_step"], checkpoint["global_epoch"]

    def save_checkpoint(self):
        # save main wav2lip model 
        os.makedirs(self.cfg.runtime.checkpoint_dir, exist_ok = True)
        ckpt_path = os.path.join(self.cfg.runtime.checkpoint_dir,
                                f"checkpoint_step{int(self.global_step)}.pth")
        opt_state = self.model_optimizer.state_dict() if self.cfg.runtime.save_optimizer_state else None
        model_state = self.model.state_dict()
        torch.save({
            "state_dict": model_state,
            "optimizer": opt_state,
            "global_step": self.global_step,
            "global_epoch": self.global_epoch
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # save disc if trained (i.e. weight > 0)
        if self.cfg.model.disc_weight > 0: 
            ckpt_path = os.path.join(self.cfg.runtime.checkpoint_dir,
                                     f"disc_checkpoint_step{int(self.global_step)}.pth")
            opt_state = self.disc_optimizer.state_dict() if self.cfg.runtime.save_optimizer_state else None
            model_state = self.disc.state_dict()
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
            pin_memory = True
        else:
            batch_sampler = RandomEntryBatchSampler(len(dataset),
                                                    batch_size = self.cfg.runtime.batch_size,
                                                    steps_per_epoch = self.cfg.runtime.eval_forward_steps)
            pin_memory = False
            
        data_loader = data_utils.DataLoader(dataset,
                                            pin_memory = pin_memory, 
                                            batch_sampler = batch_sampler)
        return data_loader
    
    def sync_loss(self, mel, gen_window):
        """
        Get SyncNet score as loss
        mel: (B, 1, 80, 16)
        gen_window: (B, 3, 5, 192, 192)
        """
        if not self.cfg.model.syncnet_ckpt: return 0 
        half_face_window = gen_window[:, :, :, gen_window.size(3)//2:]
        half_face_window = torch.cat([half_face_window[:, :, i] for i in range(half_face_window.size(2))], dim = 1) # (B, 15, 192, 96)

        if not self.cfg.model.use_SyncNetHD: # use low res SyncNet (48, 96)
            half_face_window = torch.nn.functional.interpolate(half_face_window, (48, 96)) # to use with the default (96, 48) sized SyncNet
            
        a, v = self.syncnet(mel, half_face_window)
        y = torch.ones(half_face_window.size(0), 1).float().to(device)
        cosine_sim = torch.nn.functional.cosine_similarity(a, v)
        loss = torch.nn.BCELoss()((cosine_sim.unsqueeze(1) + 1)/2., y)
        return loss

    def l1_loss(self, gen_window, gt_window):
        return torch.nn.L1Loss()(gen_window, gt_window)
        
    def log(self, log_dict):
        if self.cfg.wandb.enable:
            wandb_utils.log(log_dict)

    def log_imgs(self, input_window, gen_window, gt_window, mode = 'train', N = 3):
        if not self.cfg.wandb.enable: return 
        input_window = (input_window.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8) # (B, 5, 192, 192, 6)
        gen_window = (gen_window.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8) # (B, 5, 192, 192, 3)
        gt_window = (gt_window.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8) # (B, 5, 192, 192, 3)

        input_ref, input_in = input_window[:N, ..., 3:], input_window[:N, ..., :3]
        
        input_ref = np.concatenate(input_ref, axis = 0)[..., ::-1] # (5*B, 192, 192, 3) and swap BGR to RGB 
        input_in = np.concatenate(input_in, axis = 0)[..., ::-1]
        gen_window = np.concatenate(gen_window[:N, ...], axis = 0)[..., ::-1]
        gt_window = np.concatenate(gt_window[:N, ...], axis = 0)[..., ::-1]

        
        imgs = np.concatenate((input_ref, input_in, gen_window, gt_window), axis = -2) # along horizontal dir of image
        wandb_utils.log({f"{mode}_Images": [wandb.Image(img, caption = "Ref, Input, Generated, GT") for img in imgs],
                         "step": self.global_step})
                         
        
    def eval(self):
        if not hasattr(self, '_test_data_loader'):
            self._test_data_loader = self.get_dataloader('val')

        total_loss, sync_loss, percep_loss, l1_loss, disc_real_loss, disc_fake_loss = [], [], [], [], [], []

        for i, (input_window, indiv_mels, mel, gt_window) in enumerate(self._test_data_loader):
            mel = mel.to(device)
            input_window = input_window.to(device)
            indiv_mels = indiv_mels.to(device)
            gt_window = gt_window.to(device)

            # eval l1 loss 
            self.model.eval()
            gen_window = self.model(indiv_mels, input_window)
            l1_loss_step = self.l1_loss(gen_window, gt_window)
            l1_loss.append(l1_loss_step)

            # eval sync loss
            sync_loss_step = self.sync_loss(mel, gen_window)
            sync_loss.append(sync_loss_step)

            # eval disc loss (optional)
            if self.cfg.model.disc_weight > 0:
                self.disc.eval()
                gt_disc = self.disc(gt_window)
                gt_disc_loss = F.binary_cross_entropy(gt_disc, torch.ones((len(gt_disc), 1)).to(device))
                disc_real_loss.append(gt_disc_loss.item())
                gen_disc = self.disc(gen_window)
                gen_disc_loss = F.binary_cross_entropy(gen_disc, torch.zeros((len(gen_disc), 1)).to(device))
                disc_fake_loss.append(gen_disc_loss.item())
                percep_loss_step = F.binary_cross_entropy(gen_disc, torch.ones(len(gen_disc), 1).to(device))
                percep_loss.append(percep_loss_step)
            else:
                percep_loss_step = 0.
                percep_loss.append(percep_loss_step)

            
            tot_loss = self.cfg.model.sync_weight * sync_loss_step + \
                       self.cfg.model.disc_weight * percep_loss_step + \
                       (1 - self.cfg.model.sync_weight - self.cfg.model.disc_weight) * l1_loss_step
            total_loss.append(tot_loss.item())

            if i == 0 and self.global_step % self.cfg.runtime.checkpoint_interval < self.cfg.runtime.eval_interval: # nearest eval step to checkpoint interval 
                self.log_imgs(input_window, gen_window, gt_window, mode = 'val')
            
            
        descrip = f"Eval@Step {self.global_step}| Loss: {sum(total_loss) / len(total_loss):.6f} | L1: {sum(l1_loss) / len(l1_loss):.6f} | Sync: {sum(sync_loss) / len(sync_loss):.4f}"
        result =  {"eval_tot_loss": sum(total_loss) / len(total_loss),
                   "eval_l1_loss": sum(l1_loss) / len(l1_loss),
                   "eval_sync_loss": sum(sync_loss) / len(sync_loss), 
                   "step": self.global_step,
                   "epoch": self.global_epoch}

        if self.cfg.model.disc_weight > 0:
            descrip += f"| Percep: {sum(percep_loss) / len(percep_loss):.4f} "
            descrip += f"| disc_real: {sum(disc_real_loss) / len(disc_real_loss): .4f}"
            descrip += f"| disc_fake: {sum(disc_fake_loss) / len(disc_fake_loss): .4f}"
            result.update({"eval_percep_loss": sum(percep_loss) / len(percep_loss),
                           "eval_disc_real": sum(disc_real_loss) / len(disc_real_loss),
                           "eval_disc_fake": sum(disc_fake_loss) / len(disc_fake_loss)})
                          
        print(descrip)
        return result
        

    def train(self):
        if not hasattr(self, 'global_step'): self.global_step = 0
        if not hasattr(self, 'global_epoch'): self.global_epoch = 0

        train_data_loader = self.get_dataloader('train')

        while self.global_epoch < self.cfg.runtime.nepochs:
            tot_loss = 0.
            pgan_loss, pgan_loss_real, pgan_loss_fake = 0., 0., 0.
            sync_loss, l1_loss, disc_loss, percep_loss = 0., 0., 0., 0.
            disc_loss_real, disc_loss_fake = 0., 0.
            prog_bar = tqdm(enumerate(train_data_loader), total = self.cfg.runtime.steps_per_epoch)
            
            self.global_epoch += 1
            for step, (input_window, indiv_mels, mel, gt_window) in prog_bar:
                # input_window: (B, 6, 5, 192, 192)
                # invid_mels: (B, 5, 1, 80, 16)
                # mel: (B, 1, 80, 16)
                # gt_windown: (B, 3, 5, 192, 192)

                # >>> train wav2lip <<<
                self.model.train()
                self.model_optimizer.zero_grad()

                mel = mel.to(device)
                input_window = input_window.to(device)
                indiv_mels = indiv_mels.to(device)
                gt_window = gt_window.to(device)
                
                gen_window  = self.model(indiv_mels, input_window) # (B, 3, 5, 192, 192)
                l1_loss_step = self.l1_loss(gen_window, gt_window)
                l1_loss += l1_loss_step.item()

                if self.cfg.model.permute_disc_data:
                    idx = torch.randperm(indiv_mels.size(1), device = device)
                    gen_window = self.model(indiv_mels, input_window[:, :, idx, :, :])
                sync_loss_step = self.sync_loss(mel, gen_window)
                sync_loss += sync_loss_step
                
                if self.cfg.model.disc_weight > 0:
                    gen_disc = self.disc(gen_window)
                    percep_loss_step = F.binary_cross_entropy(gen_disc, torch.ones(len(gen_disc), 1).to(device))
                    percep_loss += percep_loss_step
                else:
                    percep_loss_step = 0.

                if self.cfg.model.patchgan_weight > 0:
                    pgan_disc = self.patchgan(gen_window)
                    pgan_loss_step = F.binary_cross_entropy(pgan_disc, torch.ones(len(pgan_disc)).to(device))
                    pgan_loss += pgan_loss_step
                else:
                    pgan_loss_step = 0.
                
                
                loss = self.cfg.model.sync_weight * sync_loss_step + \
                       self.cfg.model.disc_weight * percep_loss_step + \
                       self.cfg.model.patchgan_weight * pgan_loss_step + \
                       (1 - self.cfg.model.sync_weight - self.cfg.model.disc_weight - self.cfg.model.patchgan_weight) * l1_loss_step
                tot_loss += loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.model_optimizer.step() 

                # >>> train disc (if enabled) <<<
                if self.cfg.model.disc_weight > 0:
                    self.disc_optimizer.zero_grad()
                    # true window
                    gt_disc = self.disc(gt_window)
                    gt_disc_loss = F.binary_cross_entropy(gt_disc, torch.ones((len(gt_disc), 1)).to(device))
                    gt_disc_loss.backward()

                    # fake window
                    gen_disc = self.disc(gen_window.detach())
                    gen_disc_loss = F.binary_cross_entropy(gen_disc, torch.zeros(len(gen_disc), 1).to(device))
                    gen_disc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.disc.parameters(), 0.1)
                    self.disc_optimizer.step()
                    disc_loss_real += gt_disc_loss.item()
                    disc_loss_fake += gen_disc_loss.item()

                # >>> train pgan (if enabled) <<<
                if self.cfg.model.patchgan_weight > 0:
                    self.patchgan_optimizer.zero_grad()
                    # true window
                    gt_pgan = self.patchgan(gt_window)
                    gt_pgan_loss = F.binary_cross_entropy(gt_pgan, torch.ones(len(gt_pgan)).to(device))
                    gt_pgan_loss.backward()

                    # fake window
                    gen_pgan = self.patchgan(gen_window.detach())
                    gen_pgan_loss = F.binary_cross_entropy(gen_pgan, torch.zeros(len(gen_pgan)).to(device))
                    gen_pgan_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.patchgan.parameters(), 0.1)
                    self.patchgan_optimizer.step()
                    pgan_loss_real += gt_pgan_loss.item()
                    pgan_loss_fake += gen_pgan_loss.item()
                

                # >>> log <<<
                self.global_step += 1 
                if self.global_step % self.cfg.runtime.checkpoint_interval == 0:
                    self.save_checkpoint()
                    self.log_imgs(input_window, gen_window, gt_window, mode = 'train')

                if self.global_step % self.cfg.runtime.eval_interval == 0:
                    with torch.no_grad():
                        eval_result = self.eval()
                    self.log(eval_result)
                    
                # progbar
                descrip = f"Ep:{self.global_epoch} | Step:{self.global_step} | Loss:{loss:.4f} | L1:{l1_loss_step:.4f} | Sync:{sync_loss_step:.4f}"
                if self.cfg.model.disc_weight > 0:
                    descrip += f" | Percep:{percep_loss_step:.4f} | Real:{gt_disc_loss:.4f} | Fake:{gen_disc_loss:.4f}"
                if self.cfg.model.patchgan_weight > 0:
                    descrip += f" | pgan: {pgan_loss_step:.4f} | Real: {gt_pgan_loss:.4f} | Fake:{gen_pgan_loss:.4f}"
                prog_bar.set_description(descrip)

            #  log results
            train_result = {"step": self.global_step,
                            "epoch": self.global_epoch,
                            "train_loss": tot_loss / (step + 1),
                            "train_L1": l1_loss / (step + 1),
                            "train_sync": sync_loss / (step + 1)}
            
            if self.cfg.model.disc_weight > 0:
                train_result.update({"train_percep": percep_loss / (step + 1),
                                     "train_disc_real": disc_loss_real / (step + 1),
                                     "train_disc_fake": disc_loss_fake / (step + 1)})
            if self.cfg.model.patchgan_weight > 0:
                train_result.update({"pgan_percep": pgan_loss / (step + 1),
                                     "pgan_disc_real": pgan_loss_real / (step + 1),
                                     "pgan_disc_fake": pgan_loss_fake / (step + 1)})
                
            self.log(train_result)

            
                        
            
