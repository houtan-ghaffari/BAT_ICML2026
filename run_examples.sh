#!/bin/bash

# ==========================================
# 1. AudioSet 2M (AS2M)
# ==========================================
# python main.py -c "configs/as2m/as2m_cgp_config.yaml" --vit-size b --model-name BAT --pretrained-path /home/Projects/audio_ssl/pretrained_states/BAT_base.pt
# python main.py -c "configs/as2m/as2m_cgp_config.yaml" --vit-size b --model-name BEATs --pretrained-path /home/Projects/audio_ssl/pretrained_states/BEATs_iter3.pt
# python main.py -c "configs/as2m/as2m_cgp_config.yaml" --vit-size b --model-name EAT --pretrained-path /home/Projects/audio_ssl/pretrained_states/EAT-base_epoch10_pt.pt
# python main.py -c "configs/as2m/as2m_cgp_config.yaml" --vit-size b --model-name SSLAM --pretrained-path /home/Projects/audio_ssl/pretrained_states/sslam.pt
# python main.py -c "configs/as2m/as2m_vqt_config.yaml" ...
# python main.py -c "configs/as2m/as2m_pb_config.yaml" ...
# python main.py -c "configs/as2m/as2m_h2t_config.yaml" ...
# python main.py -c "configs/as2m/as2m_lp_config.yaml" ...
# python main.py -c "configs/as2m/as2m_lcgp_config.yaml" ...
# python main.py -c "configs/as2m/as2m_finetune_config.yaml" ...

# ==========================================
# 2. AudioSet 20k (AS20k)
# ==========================================
#python main.py -c "configs/as20k/as20k_cgp_config.yaml" --vit-size b --model-name BAT --pretrained-path /home/Projects/audio_ssl/pretrained_states/BAT_base.pt
#python main.py -c "configs/as20k/as20k_cgp_config.yaml" --vit-size b --model-name BEATs --pretrained-path /home/Projects/audio_ssl/pretrained_states/BEATs_iter3.pt
#python main.py -c "configs/as20k/as20k_cgp_config.yaml" --vit-size b --model-name EAT --pretrained-path /home/Projects/audio_ssl/pretrained_states/EAT-base_epoch10_pt.pt
#python main.py -c "configs/as20k/as20k_cgp_config.yaml" --vit-size b --model-name SSLAM --pretrained-path /home/Projects/audio_ssl/pretrained_states/sslam.pt
# ...

# ==========================================
# 3. ESC-50
# ==========================================
#python main.py -c "configs/esc50/esc50_cgp_config.yaml" --vit-size b --model-name BAT --pretrained-path /home/Projects/audio_ssl/pretrained_states/BAT_base.pt
# ...

# ==========================================
# 4. Speech Commands V2 (SCV2)
# ==========================================
#python main.py -c "configs/scv2/scv2_cgp_config.yaml" --vit-size b --model-name BAT --pretrained-path /home/Projects/audio_ssl/pretrained_states/BAT_base.pt
# ...

# ==========================================
# 5. BirdSet (HSN)
# ==========================================
#python main.py -c "configs/hsn/hsn_cgp_config.yaml" --vit-size b --model-name BAT --pretrained-path /home/Projects/audio_ssl/pretrained_states/BAT_base.pt
# ...

# ==========================================
# 6. DCASE2016 Task 2 (SED)
# ==========================================
# NOTE: VQT does not work for this
#python main.py -c "configs/dcase2016_task2/sed_cgp_config.yaml" --vit-size b --model-name BAT --pretrained-path /home/Projects/audio_ssl/pretrained_states/BAT_base.pt
# ...

# ==========================================
# 7. LibriSpeech
# ==========================================
#PYTHONHASHSEED=32 python main.py -c "configs/librispeech/ls_probe_config.yaml" --vit-size b --model-name BAT --pretrained-path /home/Projects/audio_ssl/pretrained_states/BAT_base.pt
#PYTHONHASHSEED=32 python main.py -c "configs/librispeech/ls_probe_config.yaml" --vit-size b --model-name BEATs --pretrained-path /home/Projects/audio_ssl/pretrained_states/BEATs_iter3.pt
#PYTHONHASHSEED=32 python main.py -c "configs/librispeech/ls_probe_config.yaml" --vit-size b --model-name EAT --pretrained-path /home/Projects/audio_ssl/pretrained_states/EAT-base_epoch10_pt.pt
#PYTHONHASHSEED=32 python main.py -c "configs/librispeech/ls_probe_config.yaml" --vit-size b --model-name SSLAM --pretrained-path /home/Projects/audio_ssl/pretrained_states/sslam.pt
