# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import logging
import argparse
from pathlib import Path
import os

import random
import numpy as np
import yaml
import torch
from tqdm.auto import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

from gluonts.dataset.loader import TrainDataLoader
from gluonts.dataset.split import OffsetSplitter
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.field_names import FieldName

import ANT.configs as diffusion_configs
from ANT.dataset import get_gts_dataset
from ANT.model.callback import EvaluateCallback

from ANT.sampler import DDPMGuidance, DDIMGuidance
from ANT.utils import (
    create_transforms,
    create_splitter,
    add_config_to_argparser,
    filter_metrics,
    MaskInput,
)

guidance_map = {"ddpm": DDPMGuidance, "ddim": DDIMGuidance}


def create_model(config, scheduler='linear'):
    hyperparams = getattr(diffusion_configs, config["diffusion_config"])
    hyperparams['timesteps'] = config['timesteps']
    model = TSDiff(
        **hyperparams,
        freq=config["freq"],
        use_features=config["use_features"],
        use_lags=config["use_lags"],
        normalization=config["normalization"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        lr=config["lr"],
        init_skip=config["init_skip"],
        DE = config['time_embed'],
        tau=config["tau"],
        device=config['device']
        )
    model.to(config["device"])
    return model

def evaluate_guidance(
    config, model, test_dataset, transformation, num_samples=100
):
    logger.info(f"Evaluating with {num_samples} samples.")
    results = []
    if config["setup"] == "forecasting":
        missing_data_kwargs_list = [
            {
                "missing_scenario": "none",
                "missing_values": 0,
            }
        ]
        config["missing_data_configs"] = missing_data_kwargs_list
    elif config["setup"] == "missing_values":
        misssing_data_kwargs_list = config["missing_data_configs"]
    else:
        raise ValueError(f"Unknown setup {config['setup']}")

    Guidance = guidance_map[config["sampler"]]
    sampler_kwargs = config["sampler_params"]
    if config['is_train']:
        sampler_kwargs['scale'] = config['train_scale']
    else:
        sampler_kwargs['scale'] = config["test_scale"]
        
        
    for missing_data_kwargs in missing_data_kwargs_list:
        logger.info(
            f"Evaluating scenario '{missing_data_kwargs['missing_scenario']}' "
            f"with {missing_data_kwargs['missing_values']:.1f} missing_values."
        )
        sampler = Guidance(
            model=model,
            prediction_length=config["prediction_length"],
            num_samples=num_samples,
            **missing_data_kwargs,
            **sampler_kwargs,
        )

        transformed_testdata = transformation.apply(
            test_dataset, is_train=False
        )
        test_splitter = create_splitter(
            past_length=config["context_length"] + max(model.lags_seq),
            future_length=config["prediction_length"],
            mode="test",
        )

        masking_transform = MaskInput(
            FieldName.TARGET,
            FieldName.OBSERVED_VALUES,
            config["context_length"],
            missing_data_kwargs["missing_scenario"],
            missing_data_kwargs["missing_values"],
        )
        test_transform = test_splitter + masking_transform

        predictor = sampler.get_predictor(
            test_transform,
            batch_size=1280 // num_samples,
            device=config["device"],
        )
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=transformed_testdata,
            predictor=predictor,
            num_samples=num_samples,
        )
        forecasts = list(tqdm(forecast_it, total=len(transformed_testdata)))
        tss = list(ts_it)
        evaluator = Evaluator()
        metrics, _ = evaluator(tss, forecasts)
        metrics = filter_metrics(metrics)
        results.append(dict(**missing_data_kwargs, **metrics))

    return results


def main(config, log_dir):
    random_seed = config['seed']
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)    
    
    # Load parameters
    dataset_name = config["dataset"]
    freq = config["freq"]
    context_length = config["context_length"]
    prediction_length = config["prediction_length"]
    total_length = context_length + prediction_length

    # Create model
    model = create_model(config, scheduler = config['schedule'])

    # Setup dataset and data loading
    dataset = get_gts_dataset(dataset_name)
    assert dataset.metadata.freq == freq

    if config["setup"] == "forecasting":
        training_data = dataset.train
    elif config["setup"] == "missing_values":
        missing_values_splitter = OffsetSplitter(offset=-total_length)
        training_data, _ = missing_values_splitter.split(dataset.train)

    num_rolling_evals = int(len(dataset.test) / len(dataset.train))
    num_rolling_evals = max(1,num_rolling_evals)

    transformation = create_transforms(
        num_feat_dynamic_real=0,
        num_feat_static_cat=0,
        num_feat_static_real=0,
        time_features=model.time_features,
        prediction_length=config["prediction_length"],
    )

    training_splitter = create_splitter(
        past_length=config["context_length"] + max(model.lags_seq),
        future_length=config["prediction_length"],
        mode="train",
    )

    callbacks = []
    if config["use_validation_set"]:
        transformed_data = transformation.apply(training_data, is_train=True)
        train_val_splitter = OffsetSplitter(
            offset=-config["prediction_length"] * num_rolling_evals
        )
        _, val_gen = train_val_splitter.split(training_data)
        val_data = val_gen.generate_instances(
            config["prediction_length"], num_rolling_evals
        )
        config["sampler_params"]['scale'] = config['train_scale']
        callbacks = [
            EvaluateCallback(
                context_length=config["context_length"],
                prediction_length=config["prediction_length"],
                sampler=config["sampler"],
                sampler_kwargs=config["sampler_params"],
                num_samples=config["num_samples"],
                model=model,
                transformation=transformation,
                test_dataset=dataset.test,
                val_dataset=val_data,
                eval_every=config["eval_every"],
            )
        ]
    else:
        transformed_data = transformation.apply(training_data, is_train=True)

    log_monitor = "train_loss"
    filename = dataset_name + "-{epoch:03d}-{train_loss:.3f}"

    data_loader = TrainDataLoader(
        Cached(transformed_data),
        batch_size=config["batch_size"],
        stack_fn=batchify,
        transform=training_splitter,
        num_batches_per_epoch=config["num_batches_per_epoch"],
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor=f"{log_monitor}",
        mode="min",
        filename=filename,
        save_last=True,
        save_weights_only=True,
    )

    callbacks.append(checkpoint_callback)
    callbacks.append(RichProgressBar())
    dev = int(config["device"].split(':')[1])
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=[dev],
        max_epochs=config["max_epochs"],
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        default_root_dir=log_dir,
        gradient_clip_val=config.get("gradient_clip_val", None),
    )
    
    if config['is_train']:
        logger.info(f"Logging to {trainer.logger.log_dir}")
        trainer.fit(model, train_dataloaders=data_loader)
        logger.info("Training completed.")
        if not config["use_validation_set"]:        
            best_ckpt_path = Path(trainer.logger.log_dir) / "best_checkpoint.ckpt"

            if not best_ckpt_path.exists():
                torch.save(
                    torch.load(checkpoint_callback.best_model_path)["state_dict"],
                    best_ckpt_path,
                )
    else:
        if config["use_validation_set"]:        
            best_ckpt_path = Path(trainer.logger.log_dir) / f"local_best_checkpoint_{config['test_scale']}.ckpt"
        else:
            best_ckpt_path = Path(trainer.logger.log_dir) / f"best_checkpoint.ckpt"
        
            
        best_ckpt_path = str(best_ckpt_path)
        logger.info(f"Loading {best_ckpt_path}.")
        best_state_dict = torch.load(os.path.join('saved_weights',best_ckpt_path.replace(f'_predH{config["pred_alpha"]}','')))
        model.load_state_dict(best_state_dict, strict=True)
        
        SAVE_PATH = Path(trainer.logger.log_dir) 
        SAVE_PATH = str(SAVE_PATH)
        
        if config['pred_alpha']==1:
            SAVE_PATH = os.path.join('TS_forecasting','standard_H',SAVE_PATH)
        else:
            SAVE_PATH = os.path.join('TS_forecasting','variable_H',SAVE_PATH) 
        os.makedirs(SAVE_PATH, exist_ok = True)
        SAVE_PATH = os.path.join(SAVE_PATH,f"results_{config['test_scale']}.yaml") 

        metrics = (
            evaluate_guidance(config, model, dataset.test, transformation)
            if config.get("do_final_eval", True)
            else "Final eval not performed"
        )
        
        with open(SAVE_PATH, "w") as fp:
            yaml.dump(
                {
                    "config": config,
                    "version": trainer.logger.version,
                    "metrics": metrics,
                },
                fp,
            )


if __name__ == "__main__":
    # Setup Logger
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./results/", help="Path to results dir"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100
    )
    parser.add_argument(
        "--time_embed", type=int, default=0
    )
    parser.add_argument(
        "--is_train", type=int, default=1
    )
    parser.add_argument(
        "--device", type=str, default='cuda:0',
    )
    parser.add_argument(
        "--task", type=str, default='TSF',
    )
    parser.add_argument(
        "--schedule", type=str, default='linear',
    )
    parser.add_argument('--train_scale', 
                        nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument(
        "--test_scale", type=float, default=4
    )
    parser.add_argument(
        "--tau", type=float, default=1
    )
    parser.add_argument(
        "--beta_ratio", type=float, default=1
    )
    parser.add_argument(
        "--seed", type=int, default=1
    )
    parser.add_argument(
        "--zero_enforce", type=int, default=0
    )
    parser.add_argument(
        "--pred_alpha", type=float, default=1
    )
    args, _ = parser.parse_known_args()
    

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    # Update config from command line
    parser = add_config_to_argparser(config=config, parser=parser)
    args = parser.parse_args()
    config_updates = vars(args)
    for k in config.keys() & config_updates.keys():
        orig_val = config[k]
        updated_val = config_updates[k]
        if updated_val != orig_val:
            logger.info(f"Updated key '{k}': {orig_val} -> {updated_val}")
    config.update(config_updates)
    
    config['seed'] = args.seed
    config["device"] = args.device
    config["timesteps"] = args.timesteps
    config['train_scale'] = args.train_scale
    config['test_scale'] = args.test_scale
    config["is_train"] = args.is_train
    config["tau"] = args.tau
    config['schedule'] = args.schedule
    config['beta_ratio'] = args.beta_ratio
    config['prediction_length'] = int(config['prediction_length']*config['pred_alpha'])
    config['time_embed'] = args.time_embed
    
    #------------------------------------------------------------------------------------------#
    # [1] Dataset
    args.out_dir = args.out_dir.replace('results/',f'results/{config["dataset"]}/')
    
    #------------------------------------------------------------------------------------------#
    # [2] Diffusion step Embedding (DE)
    DE = args.time_embed
    if DE:
        args.out_dir = args.out_dir.replace('results',f'results_w_DE')
    else:
        args.out_dir = args.out_dir.replace('results',f'results_wo_DE')
        
    #------------------------------------------------------------------------------------------#
    # [3] Scheduler
    scheduler = args.schedule
    assert scheduler in ['linear','sigmoid','cosine','cosine2','zero_enforce']
    
    if scheduler=='linear':
        from ANT.model import TSDiff_linear as TSDiff
    elif scheduler=='sigmoid':
        from ANT.model import TSDiff_linear as TSDiff
    elif scheduler=='cosine':
        from ANT.model import TSDiff_linear as TSDiff
    elif scheduler=='cosine2':
        from ANT.model import TSDiff_linear as TSDiff
    elif scheduler=='zero_enforce':
        from ANT.model import TSDiff_linear as TSDiff
    
    if scheduler in ['linear','zero_enforce']:
        args.out_dir = args.out_dir.replace('results',f'results_{scheduler}_0.0')
    else:
        args.out_dir = args.out_dir.replace('results',f'results_{scheduler}_{args.tau}')
        
    #------------------------------------------------------------------------------------------#
    # [4] Total diffusion steps (T)
    args.out_dir = args.out_dir.replace('results',f'results_T{args.timesteps}')
    
    #------------------------------------------------------------------------------------------#
    # [+ alpha] Various prediction horizons
    if args.pred_alpha!=1.0:
        args.out_dir = args.out_dir.replace('results',f'results_predH{args.pred_alpha}')
    #------------------------------------------------------------------------------------------#
    
    if args.is_train:
        args.out_dir = args.out_dir.replace('results',f'saved_weights/results')
    
    main(config=config, log_dir=args.out_dir)
