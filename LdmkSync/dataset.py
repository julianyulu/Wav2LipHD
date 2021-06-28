import os 
import cv2
import torch
import random
import glob
import shutil
import json 
from tqdm import tqdm
import numpy as np 
from pathos.multiprocessing import ProcessingPool as Pool 
import python_speech_features 
from utils.audio import AudioTools

import pdb 

syncnet_T = 5
syncnet_mel_step_size = 16
class Dataset(object):
    def __init__(self, config, split):
        self.config = config
        self.split = split 
        self.all_videos = self.get_ldmk_list(split)
        self.audio_tools = AudioTools(self.config.audio)
        if self.config.data.pre_calculate_mels:
            self.pre_calculate_melspecs()
        if self.config.data.random_displace_prob > 0:
            self.random_displace = RandomDisplace(self.config.data.random_displace_prob,
                                                  self.config.data.random_displace_scale)
        if self.config.data.random_rotate_prob > 0:
            self.random_rotate = RandomRotate(self.config.data.random_rotate_prob,
                                              self.config.data.random_rotate_degree,
                                              self.config.data.random_rotate_displace)
                                              

    def get_ldmk_list(self, split):
        if split == 'train':
            datafile = self.config.data.train_list
        elif split == 'val':
            datafile = self.config.data.val_list
        else:
            raise ValueError(f"{split} not supported for dataset loading.")
        filelist = []
        with open(datafile, 'r') as fp:
            for line in fp:
                line = line.strip()
                if ' ' in line: line = line.split()[0]
                filelist.append(line)
        return filelist

    def normalize_ldmk(self, ldmk_coords):
        # ldmks [68,2] array

        # eyes: 1: 37~41 2: 43~48
        eye1,eye2 = ldmk_coords[36:41], ldmk_coords[42:48]
        eye_center1 = np.mean(eye1, axis = 0)
        eye_center2 = np.mean(eye2, axis = 0)
        eye_dist =  np.sqrt(sum(eye_center1 - eye_center2)**2)
        
        # lower nose coords 32 ~ 36 
        nose_coords = ldmk_coords[31:36]
        nose_center = np.mean(nose_coords, axis = 0)

        # mouth: 49 ~ 68 
        mouth = ldmk_coords[48:68]

        # chin: 1~17
        chin = ldmk_coords[0:17]
        data = np.vstack([chin, mouth]) # (37, 2)
        data = (data -  nose_center) / float(eye_dist)
        
        return data.flatten() # (74,) 
        
    def get_window(self, ldmk_idx, ldmk_dict):
        ldmks = []
        for idx in range(int(ldmk_idx), int(ldmk_idx) + syncnet_T):
            ldmk_val = ldmk_dict.get(str(idx), None)
            if ldmk_val is None:
                return None
            else:
                ldmks.append(np.array(ldmk_val).reshape((68, 2)))
        ldmks = list(map(self.normalize_ldmk, ldmks))
        ldmks = np.array(ldmks) # (N, 74) 
        return ldmks

    def crop_audio_window(self, spec, start_frame):
        start_frame_num = int(start_frame)
        start_idx = int(80. * (start_frame_num / float(self.config.video.fps)))
        end_idx = start_idx + syncnet_mel_step_size
        return spec[start_idx : end_idx, :]

    def _wav2melspec_npy(self, wavpath):
        dirname = os.path.basename(os.path.dirname(wavpath))
        npypath = os.path.join(self.config.data.pre_calculate_dir, self.split, dirname + '.npy')
        wav = self.audio_tools.load_wav(wavpath, self.config.audio.sample_rate)
        orig_mel = self.audio_tools.melspectrogram(wav).T
        np.save(npypath, orig_mel)

    def pre_calculate_melspecs(self):
        outdir = os.path.join(self.config.data.pre_calculate_dir, self.split)
        print(f"Pre-calculating mels, save_dir: {outdir}")
        if os.path.isdir(outdir): shutil.rmtree(outdir)
        os.makedirs(outdir, exist_ok = True)
        wavs = [os.path.join(self.config.data.data_root, x, 'audio.wav') for x in self.all_videos]

        # debug: [self._wav2melspec_npy(x) for x in wavs]
        with Pool(self.config.data.pre_calculate_threads) as p:
            _ = list(tqdm(p.imap(self._wav2melspec_npy, wavs), total = len(wavs)))

    # def _wav2mfcc_npy(self, wavpath):
    #     sample_rate = self.config.audio.sample_rate
    #     dirname = os.path.basename(os.path.dirname(wavpath))
    #     npypath = os.path.join(self.config.data.pre_calculate_dir, self.split, dirname + '.npy')
    #     wav = self.audio_tools.load_wav(wavpath, sample_rate)
    #     orig_mfcc = python_speech_features.mfcc(wav, sample_rate, numcep = self.config.audio.n_mfcc) #(T, n_mfcc) 
    #     np.save(npypath, orig_mfcc)

    # def pre_calculate_mfcc(self):
    #     outdir = os.path.join(self.config.data.pre_calculate_dir, self.split)
    #     print(f"Pre-calculating mfcc, save_dir: {outdir}")
    #     if os.path.isdir(outdir): shutil.rmtree(outdir)
    #     os.makedirs(outdir, exist_ok = True)
    #     wavs = [os.path.join(self.config.data.data_root, x, 'audio.wav') for x in self.all_videos]

    #     with Pool(self.config.data.pre_calculate_threads) as p:
    #         _ = list(tqdm(p.imap(self._wav2mfcc_npy, wavs), total = len(wavs)))
                
    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        # note the idx here is dummy
        while 1:
            idx = random.randint(0, len(self) - 1)
            vidname = os.path.join(self.config.data.data_root, self.all_videos[idx])
        
            # ====  prepare ldmk frames ====
            ldmk_file = os.path.join(vidname, 'lip_ldmks.json')
            if not os.path.isfile(ldmk_file): continue
            with open(ldmk_file, 'r') as fp: ldmk_dict = json.load(fp)
            if len(ldmk_dict) < 3*syncnet_T: continue
            ldmk_ids = list(ldmk_dict.keys())

            ldmk_id = random.choice(ldmk_ids)
            ldmk_coords = self.get_window(ldmk_id, ldmk_dict) #(syncnet_T, 74)
            if ldmk_coords is None: continue

            # ldmk augmentation 
            if self.config.data.random_displace_prob > 0:
                ldmk_coords = self.random_displace(ldmk_coords)
            if self.config.data.random_rotate_prob > 0:
                ldmk_coords = self.random_rotate(ldmk_coords)
                
            # ====  prepare audio ====
                
            if self.config.data.pre_calculate_mels:
                npypath = os.path.join(self.config.data.pre_calculate_dir,
                                       self.split, 
                                       os.path.basename(vidname) + '.npy')
                if os.path.isfile(npypath):
                    orig_mel = np.load(npypath)
                else:
                    print(f"Pre-calculated {npypath} no exist ! skipping...")
                    continue
            else:
                wavpath = os.path.join(vidname, "audio.wav")
                wav = self.audio_tools.load_wav(wavpath, self.config.audio.sample_rate)
                orig_mel = self.get_melspectrogram(wav) # [T, 80]

            wrong_ldmk_id = random.choice(ldmk_ids)
            while wrong_ldmk_id == ldmk_id:
                wrong_ldmk_id = random.choice(ldmk_ids)
            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = ldmk_id
            else:
                y = torch.zeros(1).float()
                chosen = wrong_ldmk_id
                
            mel = self.crop_audio_window(orig_mel.copy(), chosen) # (16, 80)
            if (mel.shape[0] != syncnet_mel_step_size): continue
            
            #  ====== prepare final data ========
            input_ldmks = torch.FloatTensor(ldmk_coords) #(5, 74)
            mel = torch.FloatTensor(mel.T).unsqueeze(0) #(1, 80, 16)
            return input_ldmks, mel, y
        



class RandomDisplace:
    def __init__(self, prob = 0.1, max_displace = 0.01):
        """
        Add random displacement to normalized ldmks 

        prob: prob of doing this displacement 
        max_displace: maximum displacment in ratio of eye-distance 
        """
        
        self.prob = prob
        self.max_displace = max_displace

    def _random_jiter(self, x):
        """
        x: (74,)
        """
        for i in range(len(x)):
            if random.random() > self.prob:
                x[i] += random.random() * self.max_displace
        return x 

    def __call__(self, ldmk_coords):
        """
        ldmk_coords: (5, 74) [numpy]
        """
        return np.apply_along_axis(self._random_jiter, 1, ldmk_coords)

class RandomRotate:
    def __init__(self, prob = 0.1, max_angle_degree = 30, max_displace = (0, 0)):
        """
        Add random displacement to normalized ldmks 

        prob: prob of doing this displacement 
        max_displace: maximum displacment in ratio of eye-distance 
        max_displace: apply overall displacement before rotation, scale to eye-distance
        """
        
        self.prob = prob
        self.max_displace = max_displace
        self.max_angle_degree = max_angle_degree

    def _random_rotate(self, x):
        """
        x: (37, 2)
        """
        x = x.reshape((37, 2)) # (5, 37, 2)
        angel_deg = (2*random.random() - 1) * self.max_angle_degree
        angel_rad = np.radians(angel_deg)
        c,s = np.cos(angel_rad), np.sin(angel_rad)
        R = np.array(((c,s), (-s, c)))

        new_center = np.array([random.random() * z for z in self.max_displace])
        new_x = np.apply_along_axis(lambda z: np.matmul(R, z.reshape(2,1)).flatten(), 1, x)
        return new_x.flatten()

    def __call__(self, ldmk_coords):
        """
        ldmk_coords: (5, 74) [numpy]
        """
        return np.apply_along_axis(self._random_rotate, 1, ldmk_coords)



        
