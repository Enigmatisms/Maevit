# Maevit

---

​		Unofficial implementation of Vision Transformer and its variants: ViT-Lite, Compact Vision Transformer and Compact Convolution Transformer:

- [Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16x16 words: Transformers for image recognition at scale[J]. arXiv preprint arXiv:2010.11929, 2020.](https://arxiv.org/abs/2010.11929)
- [Hassani A, Walton S, Shah N, et al. Escaping the big data paradigm with compact transformers[J]. arXiv preprint arXiv:2104.05704, 2021.](https://arxiv.org/abs/2104.05704)

​		For more information, plz refer to my blog post: [Event Horizon: Vision Transformers](https://enigmatisms.github.io/2021/11/28/Vision-Transformers/).

---

## Requirements

Run the following for dependency check.

```shell
python3 -m pip install -r requirements.txt
```

| torch   | torchvision | numpy    | matplotlib | tensorboard | timm    |
| ------- | ----------- | -------- | ---------- | ----------- | ------- |
| >=1.7.0 | >=0.8.1     | >=1.18.5 | >=3.3.1    | >=2.4.0     | >=0.4.5 |

​		Plus, cuDNN is suggested, for in the main executable `train.py`, cuDNN acceleration is set.

​		CUDA is also required. Guess you won't want to fry eggs with your CPU for several days.

---

## Repo Structure

```
.
├── logs/ --- tensorboard log storage (compulsory)
├── model/ --- folder from and to which the models are loaded & stored (compulsory)
├── check_points/ --- check_points folder (compulsory)
├── train.sh --- Script for training
├── train.py --- Python main module
├── plot.sh --- quick tensorboard intialization script
└── py/ 
	 ├── CCT.py  --- Compact Convolution Transformer class
	 ├── LECosineAnnealing.py --- LECAWS lr, see my blog for more info
	 ├── LabelSmoothing.py --- Label smooting cross entropy loss
	 ├── StochasticDepth.py --- Adopted from timm
	 ├── TEncoder.py --- Tranformer encoder
	 ├── ViTLite.py --- ViT-Lite Implementaion
	 ├── configs.py --- Mixup configurations
	 ├── train_utils.py --- training utility functions.
	 └── SeqPool.py --- Sequential pooling layer
```

---

## Run the Code

​		Since `argparser` is used, some arguments must be given or they can only be loaded by defaults. In `train.sh`, some of the most-used args are provided. To run the code, therefore, run:

```sh
sudo chmod +x ./train.sh
./train.sh
```

​		Make sure you have `check_points` `model` `logs`.

​		In `py/train_utils.py`, the loading path of dataset is specified. `../dataset/` is the default `cifar-10 dataset` path. If you have no downloaded CIFAR-10 dataset, function `getCIFAR10Dataset` will download the dataset if root is set to `../datatset` and the root is empty. Once the dataset is ready, everything might just be good to go.

---

## Results

​		The tested implementation is CCT-7 transformer layers-1 x 3*3 conv kernel. The result of official counterpart is: (200 epochs - acc 94.78%).

​		The implementation of mine is: (300 epochs - training acc: roughly 100%, test acc: 94.5%), **<u>without</u>** mixup or cutmix like it does in [SHI-Labs/Compact-Transformers](https://github.com/SHI-Labs/Compact-Transformers).

​		The following image is the final version **<u>with </u>** mixup and cutmix (configuration is nquite different from the official implementation). I didn't train this from the scratch:

![](https://enigmatisms.github.io/2021/11/28/Vision-Transformers/Screenshot%20from%202021-12-07%2019-40-38.png)



