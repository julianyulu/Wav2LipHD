import os
import cv2
import dlib
import time
import json
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

try: 
    currdir = os.path.dirname(os.path.abspath(__file__))
except:
    currdir = '.'

face_detector = dlib.get_frontal_face_detector() # HOG-svm face detector
ldmk_detector = dlib.shape_predictor(currdir + '/shape_predictor_68_face_landmarks.dat')


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

    

def get_ldmk_from_image(img, visualize = False, outdir = 'ldmk_outputs'):
    """Extract ldmk from img
    img: img file path or numpy img
    """
    if type(img) == str:
        img_file = img 
        img = cv2.imread(img)
    else:
        img_file = 'numpy_img.jpg'
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 1)
    ldmks = []
    for i, rect in enumerate(rects):
        ldmk = ldmk_detector(gray, rect) # ldmk: shape obj. of dlib
        ldmk = FaceTools.ldmk_to_points(ldmk) # 2d numpy (68, 2)
        ldmks.append(ldmk)
        if visualize:
            (x, y, w, h) = FaceTools.rect_to_bbox(rect)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, f"Face #{i+1}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            for (x, y) in ldmk:
                cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
    if visualize:
        outdir_vis = os.path.join(outdir, 'visualize')
        os.makedirs(outdir_vis, exist_ok = True)
        cv2.imwrite(os.path.join(outdir_vis, os.path.basename(img_file)[:-4] + '_ldmk.jpg'), img)
        
    return ldmks # [array of size (68 x 2) s ]

def get_ldmk_from_video(video_file, resize_factor = 1.0, assert_single_face = True):
    video_stream = cv2.VideoCapture(video_file)
    fps = video_stream.get(cv2.CAP_PROP_FPS)

    ldmks = []
    k = 0
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        if not abs(resize_factor - 1) < 0.001:
            frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))
        ldmk = get_ldmk_from_image(frame)
        if assert_single_face:
            assert len(ldmk) == 1, f"Found {len(ldmk)} faces in {k} th frame of {video_file}"
        ldmks.append(ldmk) 
        k += 1
    return ldmks 


def process_fn(img_dir):
    imgs =  glob.glob(os.path.join(img_dir, '*.jpg'))
    if len(imgs)  == 0: return
    
    res = {}
    for img_file in imgs:
        num = os.path.basename(img_file).split('.')[0]
        ldmk = get_ldmk_from_image(img_file)
        if len(ldmk) == 0: continue 
        ldmk = ldmk[0][49-1: 68 + 1 - 1, :].flatten().tolist() # id: 49 ~ 68 are 20 lip ldmks (idx starts from 1)
        res[num]=ldmk

    with open(os.path.join(img_dir, 'lip_ldmks.json'), 'w') as fp:
        json.dump(res, fp)
    
if __name__ == '__main__':
    import glob
    #processed_data_dir = 'preprocessed_data/ZYDH6'
    processed_data_dir = 'preprocessed_data/mix_ZYDH6_LRS3SPK90'
    #dirs = glob.glob(processed_data_dir + '/*/*')
    dirs = glob.glob(processed_data_dir + '/*')
    res = process_map(process_fn, dirs, max_workers = 16)
    
