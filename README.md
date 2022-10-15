# MOF 
Meta-Optimized Frames for Efficient Cross-Modal Video Retrieval
----
Repository containing the code, models, data for end-to-end retrieval. 

----
### Dependencies 

Our model was developed and evaluated using the following package dependencies:
- PyTorch 1.8.0
- Transformers 4.6.0
- OpenCV 4.4.0

For more dependencies check [here](https://github.com/m-bain/frozen-in-time/blob/main/environment.yml).

### 🔧 Finetuning (benchmarks: MSR-VTT)

1. Train `python train.py --config configs/msrvtt_4f_i21k.json`

2. Test `python test.py --resume exps/models/{EXP_NAME}/{EXP_TIMESTAMP}/model_best.pth`

For finetuning a pretrained model, set `"load_checkpoint": "PATH_TO_MODEL"` in the config file.

### Datasets
We trained models on the MSR-VTT, MSVD and DiDeMo datasets. To download the datasets, refer to this [repository](https://github.com/ArrowLuo/CLIP4Clip).

### Pretrained Weights

Pretrained Weights can be found [here](https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/models/cc-webvid2m-4f_stformer_b_16_224.pth.tar).


## Acknowledgements

Our implementations use the source code from the following repositories:

* [Frozen️ in Time](https://github.com/m-bain/frozen-in-time)

* [Dataset Distillation](https://github.com/SsnL/dataset-distillation)
