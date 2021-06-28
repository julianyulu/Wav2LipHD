import os
import sys 
import glob
import cv2
import torch
import json
import dlib 
import argparse
import subprocess
import numpy as np
from tqdm import tqdm 
from scipy import signal
from omegaconf import OmegaConf
from LdmkSync.model import LdmkSync
from utils import face_detection
from utils.audio import AudioTools
import pdb

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

syncnet_T = 5
SMOOTH_WINDOW = 4
syncnet_mel_step_size = 16





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g',
                        '--gpuid',
                        default = 0,
                        help = 'gpu id (single gpu supported only)')
    parser.add_argument('-t',
                        '--max_time',
                        type = float, 
                        default = -1)
    parser.add_argument('-p',
                        '--vertical_pad',
                        default = 0,
                        help = 'vertical padding pixel number')
    parser.add_argument('-b',
                        '--batch_size',
                        default = 8)
    parser.add_argument('-i',
                        '--input_video',
                        required = True,
                        help = "input video file *.mp4")
    parser.add_argument('-v',
                        '--visualize', 
                        action = 'store_true', 
                        help = "input video file *.mp4")
    parser.add_argument('-d',
                        '--outdir',
                        default = 'temp_syncnet')
    return parser.parse_args()

class FaceTools:

    @staticmethod
    def rect_to_bbox(rect):
        """convert rect face detection by dlib to opencv (x, y, w, h) format 
        """
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return (x, y, w, h)

    @staticmethod
    def ldmk_to_points(shape, dtype = 'int'):
        """convert dlib ldmk shape obj to numpy point array (68 x 2)
        """
        coords = np.zeros((68, 2), dtype = dtype)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

class SyncInfer:
    """
    v1: single face only 
    """
    def __init__(self, config):
        self.cfg = config
        self.fps = 25
        # initialize models 
        
        self.audio_tools = AudioTools(self.cfg.audio)

        # prepare output dir 
        vfname = os.path.basename(config.infer.input_video).replace('.mp4', '')
        self.outdir = os.path.join(config.infer.outdir, vfname)
        os.makedirs(self.outdir, exist_ok=True)

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    
    def set_face_detector(self):
        return face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    def set_ldmk_model(self):
        model = LdmkSync()
        ckpt = self.cfg.infer.checkpoint
        print(f"Loading checkpoint: {ckpt} ...", end = "", flush = True)
        if torch.cuda.is_available():
            checkpoint = torch.load(ckpt, map_location = lambda storage, loc: storage)
        else:
            checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(self.device)
        for p in model.parameters(): p.requires_grad = False
        print('Done.')
        return model 

    def extract_audio_file(self):
        vfile = self.cfg.infer.input_video 
        print(f"Start processing audio file from : {vfile} ...", end="", flush = True)
        template = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'
        wavpath = os.path.join(self.outdir, 'audio.wav')
        command = template.format(vfile, wavpath)
        subprocess.call(command, shell=True)
        print("Done.")

    def extract_video_frames(self):
        """
        write out detected face 
        """
        vfile = self.cfg.infer.input_video
        max_time = self.cfg.infer.max_time
        frame_outdir = os.path.join(self.outdir, 'video_frames')
        os.makedirs(frame_outdir, exist_ok = True)
        #if os.path.exists(outdir): print(f"Folder exists {outdir}, skip extracting video frames ...")
        
        video_stream = cv2.VideoCapture(vfile)
        fps = int(video_stream.get(cv2.CAP_PROP_FPS))
        n_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_time > 0: n_frames =  int(max_time * fps)
        print(f"Extracting Video frames from {vfile}, with {n_frames}@{fps}FPS... ", end = "", flush = True)
        
        frames = []
        cnt = 0
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            frames.append(frame)
            cv2.imwrite(os.path.join(frame_outdir, f'{cnt}.jpg'), frame)
            cnt += 1
            if max_time > 0 and cnt == n_frames: break
        frames = np.array(frames)
        assert n_frames == len(frames), f"{n_frames} vs {len(frames)}"
        print("Done.")
        print(f"[Info] Video contains {n_frames} frames, duraion: {n_frames * 1./fps:.4f}s")
        self.fps = fps 
        return frames

    def get_audio_feature(self):
        print("Extracting audio features ...", end = "", flush = True)
        audio_file = os.path.join(self.outdir, 'audio.wav')
        wav = self.audio_tools.load_wav(audio_file, sr = self.cfg.audio.sample_rate)
        mel = self.audio_tools.melspectrogram(wav).T # (T, 80)
        print('Done.')
        print(f'[Info] Audio feature size {mel.shape}')
        return mel 

    def get_face_ldmk_detection(self, frames):
        n_frames = len(frames)
        vpad = self.cfg.infer.vertical_pad
        vpad = vpad if vpad > 0 else frames[0].shape[0] / 1024 * 50
        batch_size = self.cfg.infer.batch_size

        batches = [np.arange(i, min(i + batch_size, n_frames)) for i in range(0, n_frames, batch_size)]
        #face_detector = self.set_face_detector()
        face_detector = dlib.get_frontal_face_detector() # HOG-svm face detector
        ldmk_detector = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
        
        i = -1
        boxes, ldmks  = {}, {}
        print(f"Detecting face in video frames...", flush = True)
        box_window = []
        pbar = tqdm(total = n_frames)
        for fb in batches:
            #preds = face_detector.get_detections_for_batch(frames[fb])
            #for j, f in enumerate(preds):
            for j, f in enumerate(frames[fb]):
                i += 1
                pbar.update(1)
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                rects = face_detector(gray, 1)
                if len(rects) == 0: continue
                #assert len(rects) == 1
                elif len(rects) > 1:
                    print(f"{len(rects)} faces found at frame {i}, choosing the largest one ! ")
                    area = 0
                    for r in rects:
                        x, y, w, h = FaceTools.rect_to_bbox(r)
                        if w * h > area:
                            area = w * h
                            x1, y1, x2, y2 = x, y, x + w, y + h
                else: 
                    x, y, w, h = FaceTools.rect_to_bbox(rects[0])
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    
                y2 += vpad

                # smooth 
                # box_window.append([x1, x2, y1, y2])
                # if len(box_window) > SMOOTH_WINDOW:
                #     _ = box_window.pop(0)
                # smooth_box = np.mean(box_window, axis = 0, dtype = 'int32')
                # x1, x2, y1, y2 = smooth_box
                
                boxes[i] = [int(x1), int(x2), int(y1), int(y2)]
                ldmk = ldmk_detector(gray, rects[0])
                ldmk = FaceTools.ldmk_to_points(ldmk) # np array (68, 2)
                ldmks[i] = ldmk.tolist() 
                
                
        with open(os.path.join(self.outdir, 'face_boxes.json'), 'w') as fp:
            json.dump(boxes, fp)
        with open(os.path.join(self.outdir, 'face_ldmks.json'), 'w') as fp:
            json.dump(ldmks, fp)
        print("Done.")
        print(f"[Info] {len(boxes)} faces detected.")
        return boxes, ldmks
    
    def get_audio_window(self, spec, start_frame):
        # Spec: (T, 80)
        fps = self.fps 
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = int(os.path.basename(start_frame).split('.')[0])
        start_idx = int(80. * (start_frame_num / float(fps))) # audio sample rate 80Hz
        end_idx = start_idx + syncnet_mel_step_size
        if end_idx >=  len(spec):
            return None
        else:
            return spec[start_idx : end_idx, :].T  # (80, 16)

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
    
    def get_ldmk_window(self, start_id, ldmks):
        window_ldmks = [] 
        for frame_id in range(start_id, start_id + syncnet_T):
            ldmk_val = ldmks.get(frame_id, None)
            if ldmk_val is None: return None
            window_ldmks.append(np.array(ldmk_val).reshape((68, 2)))
        window_ldmks = list(map(self.normalize_ldmk, window_ldmks))
        window_ldmks = np.array(window_ldmks) #(N, 74)
        return window_ldmks 

    def calc_dist(self, feat_v, feat_a, vshift=15):
        # feat_v: (T, 512)
        # feat_a: (T, 512)
        win_size = vshift * 2 + 1
        feat_a_shift = torch.nn.functional.pad(feat_a, (0, 0, vshift, vshift)) # (T + 2*vshift , 512)
        dists = []
        for i in range(len(feat_v)):
            #d = torch.nn.functional.pairwise_distance(feat_v[[i], :].repeat(win_size, 1), feat_a_shift[i : i + win_size, :])
            d = 1 - torch.nn.CosineSimilarity()(feat_v[[i], :].repeat(win_size, 1), feat_a_shift[i : i + win_size, :])
            dists.append(d)
        return dists 

    def visualize(self):
        frame_dir = os.path.join(self.outdir, 'video_frames')
        vis_dir = os.path.join(self.outdir, 'vis_frames')
        os.makedirs(vis_dir, exist_ok = True)
        
        with open(os.path.join(self.outdir, 'face_boxes.json'), 'r') as fp:
            face_boxes = json.load(fp)
        with open(os.path.join(self.outdir, 'face_ldmks.json'), 'r') as fp:
            face_ldmks = json.load(fp)            
        with open(os.path.join(self.outdir, 'framewise_similarity.json'), 'r') as fp:
            frame_sims = json.load(fp)
        with open(os.path.join(self.outdir, 'framewise_confidence.json'), 'r') as fp:
            confs = json.load(fp) 
        
        prob_img = cv2.imread(os.path.join(frame_dir, f'{list(frame_sims.keys())[0]}.jpg'))
        height, width =  prob_img.shape[:2]
        figsize = (width // 100, height //5 // 100) # take 1/5 height of orig img as plot height 
        fig, ax = plt.subplots(figsize = figsize)
        t, y_sim, y_conf = [],[],[]
        sim_window, conf_window = [], [] # smooth window
        do_smooth = True if len(frame_sims) / self.fps > 5 else False

        # get bbox range
        temp = np.array(list(face_boxes.values()))
        bx1, by1 = temp[:, 0].min() - 20, temp[:, 2].min() - 20
        bx2, by2 = temp[:, 1].max() + 20, temp[:, 3].max() + 20
        factor  = (bx2 - bx1) * 0.2
        disp_x = (bx2 - bx1) // 2
        disp_y = (by2 - by1) // 2
        print("Start visualizing ...")
        for frame_idx in tqdm(sorted(frame_sims.keys(), key = int)):
            img = cv2.imread(os.path.join(frame_dir, f'{frame_idx}.jpg'))
            black = np.zeros((by2 - by1, bx2 - bx1, 3))
            if frame_idx in face_boxes:
                bbox = face_boxes[frame_idx]
                ldmk = face_ldmks[frame_idx]
                x1, x2, y1, y2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                #for (x,y) in ldmk: cv2.circle(img, (x, y), 2, (255, 0, 0), -1)

                ldmk = self.normalize_ldmk(ldmk).reshape((-1, 2))
                for (x,y) in ldmk: cv2.circle(black, (int(x*factor + disp_x), int(y*factor + disp_y)), 3, (255, 255, 255), -1)
                
                # for (x,y) in ldmk[0:17]: cv2.circle(black, (x - bx1, y - by1), 3, (255, 255, 255), -1)
                # for (x,y) in ldmk[48:68]: cv2.circle(black, (x - bx1, y - by1), 3, (255, 255, 255), -1)
                
                cv2.imwrite(os.path.join(vis_dir, 'black.jpg'), black)
                if frame_idx in frame_sims:
                    if do_smooth:
                        sim_window.append(frame_sims[frame_idx])
                        conf_window.append(confs[frame_idx])
                        if len(sim_window) > SMOOTH_WINDOW: _ = sim_window.pop(0)
                        sim_score = np.mean(sim_window)
                        
                        if len(conf_window) > SMOOTH_WINDOW: _ = conf_window.pop(0)
                        conf_score = np.mean(conf_window)
                    else:
                        sim_score = frame_sims[frame_idx]
                        conf_score = confs[frame_idx]    
                    y_sim.append(sim_score)
                    txt = f"Sync: {sim_score:.3f}"
                    y_conf.append(conf_score)
                    txt += f",Conf: {conf_score:.2f}"
                else:
                    txt = "N/A"
                    y_sim.append(1)
                    y_conf.append(1)                    
                cv2.putText(img, txt, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                cv2.imwrite(os.path.join(vis_dir, f'{frame_idx}.jpg'), img)
                continue
            black = cv2.resize(black, (img.shape[1], int(4*img.shape[0] //5)))
            t.append(int(frame_idx))
            ax.plot(t, y_sim, label = 'sim')
            if confs is not None:  ax.plot(t, y_conf, label = 'conf')
            ax.scatter(t[-1], y_sim[-1], c='r')
            ax.legend(fontsize='xx-small')
            ax.set_xlim([0, len(frame_sims)])
            ax.set_ylim([0, 1.0])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='y', which='major', labelsize=5)
            ax.tick_params(axis='y', which='minor', labelsize=3)
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype = np.uint8, sep = '')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            data = data[..., ::-1] # rgb to opencv bgr 
            data = cv2.resize(data, (img.shape[1], img.shape[0] // 5))
            #img = np.vstack([data, img])
            img = np.hstack([img, np.vstack([data, black])])
            cv2.imwrite(os.path.join(vis_dir, f'{frame_idx}.jpg'), img)
            plt.cla()
        cmd = f"ffmpeg -y -hide_banner -loglevel error -framerate {self.fps} -i {self.outdir}/vis_frames/%d.jpg -i {self.outdir}/audio.wav -shortest -c:v libx264 -r 25 {self.outdir}/visualize.mp4"
        print("Generating video visualization ...", end = "", flush = True)
        subprocess.call(cmd, shell = True)
        print("Done.")
                
        
        
    def infer(self):
        # extract audio & frames from video input 
        self.extract_audio_file()
        vframes = self.extract_video_frames()

        # get face detection & audio feats 
        face_bbox, face_ldmks = self.get_face_ldmk_detection(vframes)
        audio_feat = self.get_audio_feature()

        feat_a, feat_v, sims = [], [], []
        sim_dict = {}

        self.model = self.set_ldmk_model()
        self.model.eval()
        print("Inferencing A/V Syncronization ...", end = "", flush = True)
        
        for frame_idx in tqdm(sorted(face_bbox.keys(), key = int)):
            ldmk_window = self.get_ldmk_window(frame_idx, face_ldmks) # (N, 74) 
            if ldmk_window is None: continue # no Syncnet_T frames left 
            audio_window = self.get_audio_window(audio_feat, frame_idx) # (80, 16) [200 ms]
            if audio_window is None: continue 

            ldmk_window = torch.FloatTensor(ldmk_window).unsqueeze(0).to(self.device) #(1, 5, 74)
            audio_window = torch.FloatTensor(audio_window[None, ...]).unsqueeze(0).to(self.device) # (1, 1, 80, 16)
            a, v = self.model(audio_window, ldmk_window) # a: (1, 512) v: (1, 512)
            feat_a.append(a)
            feat_v.append(v)
            sim = torch.nn.functional.cosine_similarity(a, v).detach().cpu().numpy()[0]
            sims.append(sim)
            sim_dict[frame_idx] = float(sim)
        print("Done.")
        
        np.save(os.path.join(self.outdir, 'sims.npy'), sims)
        with open(os.path.join(self.outdir, 'framewise_similarity.json'), 'w') as fp:
            json.dump(sim_dict, fp)

        feat_a = torch.cat(feat_a, 0) # (T, 512)
        feat_v = torch.cat(feat_v, 0) # (T, 512)
        dists = self.calc_dist(feat_v, feat_a, vshift = 30) # T x [(31)]
        
        offset, offset_conf, fconf = calc_av_offset(dists, vshift = 30)
        fconf_dict = {int(x):float(y) for x,y in  zip(list(sorted(sim_dict.keys(), key = int)), fconf)}
        with open(os.path.join(self.outdir, 'framewise_confidence.json'), 'w') as fp:
            json.dump(fconf_dict, fp) 
        outstr = f"Video:{self.cfg.infer.input_video}\n" + \
                 f"FPS:{self.fps}\n" + \
                 f"Mean Similarity ({len(sims)} frames): {np.mean(sims): .4f}, max: {np.max(sims):.4f}, min: {np.min(sims):.4f}\n" + \
                 f"Offset: {offset} frames / {offset * 1000 / self.fps:2f} ms (confidence: {offset_conf:.3f}\n" + \
                 f"Mean frame confidence({len(fconf):.3f} frames): {np.mean(fconf[2:-2]):.3f}"

        print(outstr, flush = True)
        with open(os.path.join(self.outdir, 'result.txt'), 'w') as fp:
            fp.write(outstr)
            
        # visualize
        if self.cfg.infer.visualize:
            self.visualize()
        
def calc_av_offset(dists, vshift = 30):
    mdist = torch.mean(torch.stack(dists, 1), 1) # mean of (31, T) -> (31)
    # overall offset and offset confidence 
    minval, minidx = torch.min(mdist,0)    
    offset = vshift - minidx.cpu().numpy() 
    conf = torch.median(mdist) - minval

    # per frame distance (offset corrected) 
    fdist = torch.stack([dist[minidx] for dist in dists])

    #fdist = torch.nn.functional.pad(fdist, (3, 3))

    # fconf = np.median(torch.stack(dists, 0).cpu().numpy(), -1) - fdist.cpu().numpy()
    # fconf = signal.medfilt(fconf, kernel_size = 9)
    
    fconf = torch.median(mdist) - fdist
    fconf = signal.medfilt(fconf.cpu(), kernel_size = 9)
    return offset, conf, fconf

if __name__ == '__main__':
    args  = parse_args()
    print(args.gpuid)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    
    checkpoint = "exp/LdmkSync/ZYDH6LRS90_BC256_LR1e-3_MAV4.0_noLdmkAug/checkpoint_step99000.pth"
    config_file = 'SyncNetHD/base_config.yaml'
    config = OmegaConf.load(config_file)
    config.infer = args.__dict__
    config.infer.checkpoint = checkpoint
    print("="*30)
    print(OmegaConf.to_yaml(config))
    print("="*30)
    
    infer_op = SyncInfer(config)
    infer_op.infer()
    #infer_op.visualize()
    
