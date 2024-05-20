import pandas as pd
import matplotlib.pyplot as plt
import torch
import math
import torch.nn.functional as F
import numpy as np
from ANT.utils import extract
from statsmodels.tsa.stattools import acf
import json


def read_json(PATH):
    ts_list = []
    norm_ts_list = []

    with open(PATH, 'r') as file:
        for line in file:
            data = json.loads(line)
            ts = np.array(data['target'])
            ts_list.append(ts)
            norm_ts_list.append(ts/np.mean(np.abs(ts)))
            
    return ts_list, norm_ts_list


def linear_beta_schedule(timesteps,min_val,max_val):
    return torch.linspace(min_val, max_val, timesteps)


def get_scheduler_utils(T,betas):
    #-----------------------------------------#
    #betas = reversed(betas)
    #betas =torch.ones_like(betas)*0.0001
    #-----------------------------------------#
    sqrt_one_minus_beta = torch.sqrt(1.0 - betas)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(
        alphas_cumprod[:-1], (1, 0), value=1.0
    )
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(
        1.0 - alphas_cumprod
    )
    posterior_variance = (
        betas
        * (1.0 - alphas_cumprod_prev)
        / (1.0 - alphas_cumprod)
    )
    return sqrt_one_minus_beta, alphas, alphas_cumprod,sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance


def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(
        sqrt_alphas_cumprod, t, x_start.shape
    )
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return (
        sqrt_alphas_cumprod_t * x_start
        + sqrt_one_minus_alphas_cumprod_t * noise
    )
    

def return_acf(ts, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,T=100):
    acf_list = []
    for t_idx in range(0, T, 1):
        t = torch.LongTensor([t_idx])    
        x_noisy = q_sample(ts, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        acf_values = acf(x_noisy.detach().numpy(), fft=False, nlags=len(x_noisy)-1)
        acf_list.append(acf_values)
    return np.array(acf_list)
   
def cosine_schedule(t, start = 0, end = 1, tau = 1, clamp_min = 1e-9):
    t = torch.FloatTensor([t])
    v_start = math.cos(start * math.pi / 2) ** (2 * tau)
    v_end = math.cos(end * math.pi / 2) ** (2 * tau)
    gamma = torch.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
    gamma = (v_end - gamma) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 5.)

def sigmoid(x):
    return 1/(1+torch.exp(-x))

def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    t = torch.FloatTensor([t])
    v_start = torch.tensor(start / tau)
    v_end = torch.tensor(end / tau)
    v_start = sigmoid(v_start)
    v_end = sigmoid(v_end)
    A = -sigmoid((t * (end - start) + start) / tau) + v_end 
    B = (v_end - v_start)
    gamma = A/B
    return gamma.clamp_(min = clamp_min, max = 5.)

def cosine_schedule_beta(T,tau,min_val,max_val):
    alphas_cumprod = torch.cat([cosine_schedule(t= i/T,tau = tau) for i in range(1,T+1)])
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1)
    alphas = alphas_cumprod / alphas_cumprod_prev
    betas = 1 - alphas
    betas = (betas-betas.min())/(betas.max()-betas.min())
    betas = betas * (max_val - min_val) + min_val
    return betas

def sigmoid_schedule_beta(T,tau,min_val,max_val):
    alphas_cumprod = torch.cat([sigmoid_schedule(t= i/T, tau = tau) for i in range(1,T+1)])
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1)
    alphas = alphas_cumprod / alphas_cumprod_prev
    betas = 1 - alphas
    betas = (betas-betas.min())/(betas.max()-betas.min())
    betas = betas * (max_val - min_val) + min_val
    return betas
    
def return_betas(scheduler = 'cosine', tau = 2.0, T = 75,
                 beta_0=0.0001, beta_T=0.1):
    if scheduler=='linear':
        betas = linear_beta_schedule(T,beta_0,beta_T)
    elif scheduler=='cosine':
        betas = cosine_schedule_beta(T,tau,beta_0,beta_T)
    elif scheduler=='sigmoid':
        betas = sigmoid_schedule_beta(T,tau,beta_0,beta_T)
    return betas

def preprocess_data(x):
    # any functions!
    x = x/np.mean(np.abs(x))
    x = torch.FloatTensor(x)
    return x

def pad_acf_data(TS_acf_variable):
    temp = []
    length_list = []
    for ts_acf in TS_acf_variable:
        ts_acf = np.stack(ts_acf)
        temp.append(ts_acf)
        length_list.append(ts_acf.shape[1])
    med_length = int(np.mean(length_list))

    for idx, ts in enumerate(temp):
        T, length = ts.shape
        if length>med_length:
            temp[idx] = ts[:,:med_length]
        elif length<med_length:
            temp[idx] = np.pad(ts, ((0, 0), (0, med_length-length)), mode='constant')
            
    TS_acf_variable = np.stack(temp)
    return TS_acf_variable

def get_IAAT_list(TS_acf):
    T = TS_acf.shape[1]
    IAAT = []
    for t in range(T):
        acf_mean = np.mean(TS_acf[:,t,:], axis=0)
        iaat = 1+(np.abs(acf_mean).sum()*2)-2
        IAAT.append(iaat)
    IAAT = np.array(IAAT)
    scaled_IAAT = (IAAT-IAAT.min())/(IAAT.max()-IAAT.min())
    return IAAT, scaled_IAAT
        
def calculate_auc(y):
    T = len(y)
    base = np.linspace(1,0,T)
    area = 0
    for i in range(T-1):
        if i==0:
            y_target = y[i+1]
            y_gt = base[i+1]
            area += 0.5*(1/T)*(np.abs(y_target-y_gt))
        elif i==T-2:
            y_target = y[i]
            y_gt = base[i]
            area += 0.5*(1/T)*(np.abs(y_target-y_gt))
        else:
            y_target1 = y[i]
            y_gt1 = base[i]
            y_target2 = y[i+1]
            y_gt2 = base[i+1]
            area += 0.5*(1/T)*(np.abs(y_target1-y_gt1)+np.abs(y_target2-y_gt2))
    return area
            
                
def calculate_ANT_score(IAAT, scaled_IAAT):
    T = len(IAAT)
    lambda_linear = calculate_auc(scaled_IAAT)
    lambda_noise = 1+IAAT[-1]/IAAT[0]
    lambda_steps = 1+1/T
    return lambda_linear*lambda_noise*lambda_steps    