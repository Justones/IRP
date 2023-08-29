# Exploring and Evaluating Image Restoration Potential in Dynamic Scenes (CVPR22)


[[paper link](https://arxiv.org/pdf/2203.11754.pdf)]  & [[Dataset link](https://drive.google.com/file/d/1YhjBCBBFvRlSDiCvVrM-smkNlgaejX3N/view?usp=sharing)]

**Authors**: Cheng Zhang, Shaolin Su, Yu Zhu, Qingsen Yan, Jinqiu Sun, Yanning Zhang;

**Abstract**: 
> In dynamic scenes, images often suffer from dynamic blur due to superposition of motions or low signal-noise ratio resulted from quick shutter speed when avoiding motions. Recovering sharp and clean result from the captured images heavily depends on the ability of restoration methods and the quality of the input. Though existing research on image restoration focuses on developing models for obtaining better restored results, less have studied to evaluate how and which input image leads to superior restored quality. In this paper, to better study an image's potential value that can be explored for restoration, we propose a novel concept, referring to image restoration potential (IRP). Specifically, We first establish a dynamic scene imaging dataset containing composite distortions and applied image restoration processes to validate the rationality of the existence to IRP. Based on this dataset, we investigate into several properties of IRP and propose a novel deep model to accurately predict IRP values. By gradually distilling and selective fusing the degradation features, the proposed model shows its superiority in IRP prediction. Thanks to the proposed model, we are then able to validate how various image restoration related applications are benefited from IRP prediction. We show the potential usages of IRP as a filtering principle to select valuable frames, an auxiliary guidance to improve restoration models, and also an indicator to optimize camera settings for capturing better images under dynamic scenarios.


### Dependencies and Installation  
- Python3, NVIDIA GPU + Anaconda
- torch, torchvision, numpy, pillow, opencv-python, pretrainedmodels, glob, guided_filter_pytorch


-      -      -
### Datasets

- Synthetic data: The synthetic data can be found in the above link, including training images, test images, and IRP labels.
- Real data: Any RGB image from the real world can be employed as the input.

-     -     -

### Train

1. Prepare training data
- download IRPDataset and store it on your computer ;
- modify the `config.data_path` configuration within the `train.py` file to match the current directory path.
2. run commands
```
python train.py
```
-    -     -


### Test


Update the `config.weights_path` configuration in the `test.py` file with the file path of your trained model.

```
python test.py
```
-      -     -

### Citation

```
@inproceedings{zhang2022exploring,
  title={Exploring and evaluating image restoration potential in dynamic scenes},
  author={Zhang, Cheng and Su, Shaolin and Zhu, Yu and Yan, Qingsen and Sun, Jinqiu and Zhang, Yanning},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2067--2076},
  year={2022}
}
```
