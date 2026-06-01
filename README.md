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

2. Install PyTorch and TorchAudio according to your system specifications from the [official PyTorch website](https://pytorch.org/). Do not forget to add the torchaudio as well, For example:
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

* **AudioSet**: use the `download_audioset.py` to download the AudioSet. Inside the file, set `download_unbalanced=True` to download the AS-2M as well. By default, it only downloads the AS-20k and the eval set. Also change the `target_dir` to save the dataset in your preferred path. Beware that downloading AS-2M requires ~1.2 TB of space. The metadata of the dataset (csv files) are available in this repo. 
---

## Self-Supervised Pretraining

---

## Downstream Tasks

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
