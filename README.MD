# Efficient Text-driven Motion Generation via Latent Consistency Training (MLCT)

MLCT focus on conditional consistency training pipeline within motion latent representation. Our work achieves state-of-the-art motion performance in just **5-10 NFE**, saving over **80%** inference time compared to the fastest previous models.


## ⚡ Quick Start

### 1. Conda environment

```
conda create python=3.9 --name mlct
conda activate mlct
```

Install the packages in `requirements.txt` and Pytorch.

```
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

We test our code on Python 3.9.17 and PyTorch 1.11.0+cu113.

### 2. Dependencies

Follow the [MLD](https://github.com/ChenFengYe/motion-latent-diffusion) and run the script to download the dependent material:

```
bash prepare/download_smpl_model.sh
bash prepare/prepare_clip.sh
bash prepare/download_t2m_evaluators.sh
```

Downloaded dependent materials are placed in the `deps` folder.

### 3. Pre-train model

Run the script to download the pre-train model

```
bash prepare/download_pretrained_models.sh
```


### 4. Data

Please refer to [HumanML3D](https://github.com/EricGuo5513/HumanML3D) for text-to-motion dataset setup.
We will provide instructions for other datasets soon. [GraphMotion](https://github.com/jpthu17/GraphMotion) offers a more convenient way to download.

## 💻 Training Guidance
### 1. Ready to train motion autoencoder

All the parameters are report in `configs/{dataset_name}/{dataset_name}_mae_train.yaml`.

Then, run the following command, as example in HumanML:

```
python -m tools.train_mae --cfg config/humanml/humanml_mae_train.yaml
```

### 2. Ready to train motion consistency model via consistency training

All the parameters are report in `configs/{dataset_name}/{dataset_name}_mlct_train.yaml`. The second phase of training requires checking the `motion_ae.pretrain_dir` and `motion_ae.pretrain_model_name` of the motion autoencoder.

Then, run the following command, as example in HumanML:

```
python -m tools.train_mlct --cfg configs/humanml/humanml_mlct_train.yaml
```

## 💻 Evaluating Guidance
Please first put the tained model checkpoint the `motion_ae.pretrain_dir`, `motion_ae.pretrain_model_name`, `diffusion.pretrain_dir` and `diffusion.pretrain_model_name` in `configs/humanml/humanml_test.yaml`.

Then, run the following command:

```
python -m tools.test --cfg configs/humanml/humanml_test.yaml
```

## Acknowledgments

Thanks to [TEMOS](https://github.com/Mathux/TEMOS), [HumanML3D](https://github.com/EricGuo5513/HumanML3D), [MLD](https://github.com/ChenFengYe/motion-latent-diffusion), and [GraphMotion](https://github.com/jpthu17/GraphMotion), our code is partially borrowing from them.

## License

This code is distributed under an [MIT LICENSE](LICENSE).
