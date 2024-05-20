# ANT: Adaptive Noise Scheduler for Time Series Diffusion Model

<br>

## Installation

Create a conda environment

```
conda create --name ANT --yes python=3.8 && conda activate ANT

pip install --editable "."
```

<br>

# Three Downstream Tasks

Example)

- Dataset: `Wikipedia`
- Scheduler: `Cos(T=75, tau=2.0)`

<br>

## 1) TS Forecasting

**a) Standard horizon ($H$)**

- Train

```sh
python bin/train_model.py -c configs/train_tsdiff/train_wiki2000_nips.yaml --schedule cosine --tau 2.0 --timesteps 75 --time_embed 1 --is_train 1 --train_scale 4.0,8.0,16.0,32.0
```

<br>

- Test

```sh
python bin/train_model.py -c configs/train_tsdiff/train_wiki2000_nips.yaml --schedule cosine --tau 2.0 --timesteps 75 --time_embed 1 --is_train 0 --test_scale 8.0 --train_scale 999
```

<br>

**b) Variable horizons ($\alpha \cdot H$)**

- Train ( where $\alpha=2$)

```sh
python bin/train_model.py -c configs/train_tsdiff/train_wiki2000_nips.yaml --schedule cosine --tau 2.0 --timesteps 75 --pred_alpha 2.0 --time_embed 1 --is_train 1 --train_scale 4.0,8.0,16.0,32.0
```

- Test

```sh
python bin/train_model.py -c configs/train_tsdiff/train_wiki2000_nips.yaml --schedule cosine --tau 2.0 --timesteps 75 --pred_alpha 2.0 --time_embed 1 --is_train 0 --test_scale 16.0 --train_scale 999
```

<br>

## 3-2) TS Refinement

Load pretrained weights trained from **3-1) TS Forecasting**

```sh
python bin/refinement_experiment.py -c configs/refinement/wiki2000_nips-linear.yaml --timesteps 75 --schedule cosine --tau 2.0 --time_embed 1 --ckpt saved_weights/results_T75_cosine_2.0_w_DE/wiki2000_nips/lightning_logs/version_0/checkpoints/last.ckpt
```

<br>

## 3-3) TS Generation

Load pretrained weights trained from **3-1) TS Forecasting**

```sh
python bin/tstr_experiment.py -c configs/tstr/wiki2000_nips.yaml --ckpt saved_weights/results_T75_cosine_2.0_w_DE/wiki2000_nips/lightning_logs/version_0/local_best_checkpoint_4.0.ckpt --schedule cosine --timesteps 75 --tau 2.0 --time_embed 1
```


# Acknowledgement

We appreciate the following github repositories for their valuable code base & datasets:

https://github.com/amazon-science/unconditional-time-series-diffusion