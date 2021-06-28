#/bin/bash

export CUDA_VISIBLE_DEVICES=6

#python infer_SyncNetHD.py  -v -g 6 -i ../Wav2LipHD/examples/xiaoqian_1080_1080_38sec.mp4 -s 96  -d output_syncnetDF_new
# python infer_SyncNetHD.py  -v -g 6 -i ../Wav2LipHD/examples/xiaoqian_1080_1080_38sec.mp4 -s 192  -d output_syncnetHD




python infer_SyncNetHD.py  -v -g 6 -i ../Wav2LipHD/examples/real_person_compare/tts_lm_0522_cm.qv2.0_04_51-0_05_04.mp4 -s 96  -d output_syncnetDF_new

python infer_SyncNetHD.py  -v -g 6 -i ../Wav2LipHD/examples/real_person_compare/tts_lm_0522_cm.qv2.0_04_51-0_05_04.mp4 -s 192  -d output_syncnetHD

#python infer_SyncNetHD.py  -v -g 6 -i /data/julianlu/Data/DigitalHuman/eval_product_videos/zhuiyi_1.mp4 -s 96  -d output_syncnetDF_new
#python infer_SyncNetHD.py  -v -g 6 -i /data/julianlu/Data/DigitalHuman/eval_product_videos/zhuiyi_2.mp4 -s 96  -d output_syncnetDF_new


# python infer_SyncNetHD.py  -v -g 6 -i /data/julianlu/Data/DigitalHuman/eval_product_videos/zhuiyi_1.mp4 -s 192  -d output_syncnetHD
# python infer_SyncNetHD.py  -v -g 6 -i /data/julianlu/Data/DigitalHuman/eval_product_videos/zhuiui_2.mp4 -s 192  -d output_syncnetHD



# python infer_SyncNetHD.py  -v -g 6 -i /data/julianlu/Experiment/Wav2LipHD/examples/lrs3_luoK0kTx0tU.mp4 -s 192  -d output_syncnetHD

