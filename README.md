# Better Audio Transformer
This is the official pytorch code for BAT: Better Audio Transformer Guided by Convex Gated Probing, ICML 2026.

---

## Installation

This project was developed using **Python 3.11**, **PyTorch 2.8**, and **TorchAudio 2.8**. However, it is expected to work seamlessly with more recent versions of Python and PyTorch.

1. I highly recommend creating an isolated Conda environment, for example:
```bash
conda update -n base conda
conda create -n bat python=3.11
conda activate bat
```

2. Install PyTorch and TorchAudio according to your system specifications from the [official PyTorch website](https://pytorch.org/). Do not forget to add the torchaudio as well, for example:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

3. Install any other missing packages using pip, but these are necessary:
```bash
pip install librosa datasets==3.6.0 sed_eval jiwer scikit-learn pandas tqdm scipy
```

---


## Data
* Download these datasets manually:
  * [Speech Commands](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)

  * [ESC-50](https://github.com/karolpiczak/ESC-50)

  * [DCASE 2016 Task 2 (Sound Event Detection)](https://dcase.community/challenge2016/task-sound-event-detection-in-synthetic-audio)

* The Librispeech and High Sierra Nevada (BirdSet) are downloaded automatically the first time you run their downstream tasks.

* **AudioSet**: use the `download_audioset.py` to download the AudioSet. Inside the file, set `download_unbalanced=True` to download the AS-2M as well. By default, it only downloads the AS-20k and the eval set. Also change the `target_dir` to save the dataset in your preferred path. Beware that downloading AS-2M requires ~1.2 TB of space. Download metadata of the AudioSet (csv files and ontology) from these links:
  * http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv
  * http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv
  * http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv
  * download the ontology.json from here: https://github.com/audioset/ontology
  
---

## Self-Supervised Pretraining
Change any desired configuration inside the `configs/ssl.yaml`, and run the following with the desired number of GPUs: 

```bash
torchrun --nproc_per_node=4 --master_port=29500 pretrain.py -c configs/ssl.yaml
```

It also works with one GPU (`--nproc_per_node=1`), and you can simulate multi-GPU run by increasing the `grad_accumulation_steps` in the config file.

---

## Downstream Tasks

For any downstream task, you simply need to pass the desired config for a probing method or finetuning, alongside the model size and name. For example, the commands below run the CGP on AudioSet-20k for each model:

```bash
python main.py -c "configs/as20k/as20k_cgp_config.yaml" --vit-size b --model-name BAT --pretrained-path /home/Projects/audio_ssl/pretrained_states/BAT_base.pt
python main.py -c "configs/as20k/as20k_cgp_config.yaml" --vit-size b --model-name BEATs --pretrained-path /home/Projects/audio_ssl/pretrained_states/BEATs_iter3.pt
python main.py -c "configs/as20k/as20k_cgp_config.yaml" --vit-size b --model-name EAT --pretrained-path /home/Projects/audio_ssl/pretrained_states/EAT-base_epoch10_pt.pt
python main.py -c "configs/as20k/as20k_cgp_config.yaml" --vit-size b --model-name SSLAM --pretrained-path /home/Projects/audio_ssl/pretrained_states/sslam.pt
```

---

## Pretrained Weights

You can download the BAT ViT-base architecture from this link: [BAT_base.pt](https://drive.google.com/uc?export=download&id=1fJ3FA4HC9fICQkpkk5AmZOd4O2sseg30)

## Hugging Face

We release the pretrained BAT ViT-B/16 encoder on Hugging Face:

**https://huggingface.co/lrauch/BAT-vit-b16-pretrainedAS2M**

There are two main ways to use the release.

### 1. Load the model directly

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "lrauch/BAT-vit-b16-pretrainedAS2M",
    trust_remote_code=True,
).eval()

features = torch.randn(2, 1, 1024, 128)

with torch.no_grad():
    outputs = model(input_features=features)
```

### 2. Download only the checkpoint weights

If you want to integrate the pretrained encoder weights into your own codebase (or here), download the raw `model.safetensors` file:

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

weights_path = hf_hub_download(
    repo_id="lrauch/BAT-vit-b16-pretrainedAS2M",
    filename="model.safetensors",
)

state_dict = load_file(weights_path, device="cpu")
print(state_dict.keys())
```

---

## Citation

If this code was helpful to your research, kindly consider citing this paper:

```bibtex
@misc{ghaffari2026batbetteraudiotransformer,
      title={BAT: Better Audio Transformer Guided by Convex Gated Probing}, 
      author={Houtan Ghaffari and Lukas Rauch and Christoph Scholz and Paul Devos},
      year={2026},
      eprint={2602.16305},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2602.16305}, 
}
```
