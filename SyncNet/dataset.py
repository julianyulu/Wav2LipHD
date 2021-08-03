import os 
import cv2
import torch
import random
import glob
import shutil
from tqdm import tqdm
import numpy as np 
from pathos.multiprocessing import ProcessingPool as Pool 
import python_speech_features 
from utils.audio import AudioTools

import pdb 

syncnet_T = 5

class Dataset(object):
    def __init__(self, config, split):
        self.config = config
        self.split = split 
        self.all_videos = self.get_image_list(split)
        self.audio_tools = AudioTools(self.config.audio)
        if self.config.data.pre_calculate_mfcc:
            self.pre_calculate_mfcc()
        
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
            #img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) # read gray scale 
            if img is None:
                return None
            try:
                img = img[img.shape[0]//2:, :]
                img = cv2.resize(img, (self.config.video.img_size, self.config.video.img_size))
            except Exception as e:
                return None 

            window.append(img)
        return window

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = 4 * start_frame_num # audio: 10ms, video 40ms (25 fps)
        end_idx = start_idx + 20 # 20 * 10ms = 200ms = 40ms * 5
        if end_idx >= len(spec):
            return None
        else:
            return spec[start_idx : end_idx, :]

    
    def _wav2mfcc_npy(self, wavpath):
        sample_rate = self.config.audio.sample_rate
        dirname = os.path.basename(os.path.dirname(wavpath))
        npypath = os.path.join(self.config.data.pre_calculate_dir, self.split, dirname + '.npy')
        wav = self.audio_tools.load_wav(wavpath, sample_rate)
        orig_mfcc = python_speech_features.mfcc(wav, sample_rate, numcep = self.config.audio.n_mfcc) #(T, n_mfcc) 
        np.save(npypath, orig_mfcc)

    
    def pre_calculate_mfcc(self):
        outdir = os.path.join(self.config.data.pre_calculate_dir, self.split)
        print(f"Pre-calculating mfcc, save_dir: {outdir}")
        if os.path.isdir(outdir): shutil.rmtree(outdir)
        os.makedirs(outdir, exist_ok = True)
        wavs = [os.path.join(self.config.data.data_root, x, 'audio.wav') for x in self.all_videos]

        with Pool(self.config.data.pre_calculate_threads) as p:
            _ = list(tqdm(p.imap(self._wav2mfcc_npy, wavs), total = len(wavs)))
                
    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        # note the idx here is dummy
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = os.path.join(self.config.data.data_root, self.all_videos[idx])
            # ====  prepare video frames ==== 
            img_names = list(glob.glob(os.path.join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T: continue

            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self.get_window(img_name) 
            window = self.read_window(window_fnames) #(5, 240, 240)
            if window is None: continue
            input_frames = np.asarray(window) / 255. 
            input_frames = input_frames.transpose(3, 0, 1, 2) # (3, 5, 240, 240)
            del window
            
            # ====  prepare audio ====
            if self.config.data.pre_calculate_mfcc:
                npypath = os.path.join(self.config.data.pre_calculate_dir,
                                       self.split, 
                                       os.path.basename(vidname) + '.npy')
                if os.path.isfile(npypath):
                    orig_mfcc= np.load(npypath) #(T, n_mfcc)
                else:
                    print(f"Pre-calculated {npypath} no exist ! skipping...")
                    continue
            else:
                wavpath = os.path.join(vidname, "audio.wav")
                wav = self.audio_tools.load_wav(wavpath, self.config.audio.sample_rate)
                orig_mfcc = python_speech_features.mfcc(wav, self.config.audio.sample_rate, numcep = self.config.audio.n_mfcc) # (T, n_mfcc)

                        
            mfcc = self.crop_audio_window(orig_mfcc.copy(), chosen) # (20,13)
            if mfcc is None: continue 
            
            #  ====== prepare final data ========
            input_frames = torch.FloatTensor(input_frames) #(5, 240, 240)
            mfcc = torch.FloatTensor(mfcc.T).unsqueeze(0) #(1, 13, 20)
            return input_frames, mfcc, y
        
