import os 
import cv2
import torch
import random
import glob
import shutil
import numpy as np
from tqdm import tqdm
import audiomentations as am 
from utils.audio import AudioTools
from pathos.multiprocessing import ProcessingPool as Pool

import pdb 

syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, config, split):
        self.config = config
        self.split = split 
        self.audio_tools = AudioTools(self.config.audio)
        self.all_videos = self.get_image_list(split)
        if self.config.data.audio_augment:
            self.augment_op = am.Compose([
                am.AddGaussianNoise(p = self.config.data.gaussian_noise_prob),
                am.FrequencyMask(p = self.config.data.freq_mask_prob),
                am.PitchShift(min_semitones = -5,
                              max_semitones = 5,
                              p = self.config.data.pitch_shift_prob)
                ])

        if self.config.data.pre_calculate_mels:
            self.pre_calculate_melspecs()
                    
    def get_image_list(self, split):
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

    def get_frame_id(self, frame):
        return int(os.path.basename(frame).split('.')[0])
    
    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = os.path.dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = os.path.join(vidname, '{}.jpg'.format(frame_id))
            if not os.path.isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None 
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (self.config.video.img_size, self.config.video.img_size))
            except Exception as e:
                return None 

            window.append(img)
        return window

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        if type(start_frame) == str:
            start_frame_num = self.get_frame_id(start_frame)
        else:
            start_frame_num = start_frame
        start_idx = int(80. * (start_frame_num / float(self.config.video.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)
        mels = np.asarray(mels)
        return mels
    
    def _wav2melspec_npy(self, wavpath):
        dirname = os.path.basename(os.path.dirname(wavpath))
        npypath = os.path.join(self.config.data.pre_calculate_dir, self.split, dirname + '.npy')
        wav = self.audio_tools.load_wav(wavpath, self.config.audio.sample_rate)            
        orig_mel = self.audio_tools.melspectrogram(wav).T
        np.save(npypath, orig_mel)
    
    def pre_calculate_melspecs(self):
        outdir = os.path.join(self.config.data.pre_calculate_dir, self.split)
        if os.path.isdir(outdir): shutil.rmtree(outdir)
        os.makedirs(outdir, exist_ok = True)
        wavs = [os.path.join(self.config.data.data_root, x, 'audio.wav') for x in self.all_videos]
        print(f"Pre-calculating mels, save_dir: {outdir}")
        with Pool(self.config.data.pre_calculate_threads) as p:
            _ = list(tqdm(p.imap(self._wav2melspec_npy, wavs), total = len(wavs)))
            
    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = os.path.join(self.config.data.data_root, self.all_videos[idx])

            # ====  prepare video frames ==== 
            img_names = list(glob.glob(os.path.join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T: continue

            # get input(mask_window) & target(gt_window)  frames
            img_name = random.choice(img_names)
            window_fnames = self.get_window(img_name) 
            window = self.read_window(window_fnames) #(5, 192, 192, 3)
            if window is None: continue
            gt_window = np.transpose(np.asarray(window) / 255., (3, 0, 1, 2))
            mask_window = gt_window.copy() # [3, 5, 192, 192]
            mask_window[:, :, mask_window.shape[2]//2:, :] = 0.
            del window, window_fnames 
        
            # get ref frames 
            ref_img_name = random.choice(img_names)
            while ref_img_name == img_name:
                ref_img_name = random.choice(img_names)
            ref_window_fnames = self.get_window(ref_img_name)
            ref_window = self.read_window(ref_window_fnames) #(5, 192, 192, 3)
            if ref_window_fnames is None: continue
            ref_window = np.transpose(np.asarray(ref_window) / 255., (3, 0, 1, 2))

            # combine mask_windown & ref_window into input_window 
            input_window = np.concatenate([mask_window, ref_window], axis = 0) # [6, 5, 192, 192]
            del ref_window 
            
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
                
                if self.config.data.audio_augment and \
                   len(wav) / self.config.audio.sample_rate < 30:
                    self.augment_op(wav, self.config.audio.sample_rate)
                    
                orig_mel = self.audio_tools.melspectrogram(wav).T # [T, 80]
        
            mel = self.crop_audio_window(orig_mel.copy(), img_name) # (16, 80)
            if (mel.shape[0] != syncnet_mel_step_size): continue
            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name) # (5, 80, 16)
            if indiv_mels is None: continue
            del orig_mel
            
            #  ====== prepare final data ========
            input_window = torch.FloatTensor(input_window) # (6, 5, 192, 192)
            gt_window = torch.FloatTensor(gt_window) # (3, 5, 192, 192)
            mel = torch.FloatTensor(mel.T).unsqueeze(0) # (1, 80, 16)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1) # (5, 1, 80, 16)
            
            return input_window, indiv_mels, mel, gt_window
        
