# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import partial
import random
import math
import logging
import argparse
from pathlib import Path
import os
import yaml
import torch
import numpy as np
from tqdm.auto import tqdm

from gluonts.mx import DeepAREstimator, TransformerEstimator
from gluonts.evaluation import Evaluator
from gluonts.dataset.loader import TrainDataLoader
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.time_feature import (
    get_lags_for_frequency,
    time_features_from_frequency_str,
)
from gluonts.dataset.split import slice_data_entry
from gluonts.transform import AdhocTransform, Chain

from uncond_ts_diff.utils import (
    ScaleAndAddMeanFeature,
    ScaleAndAddMinMaxFeature,
    GluonTSNumpyDataset,
    create_transforms,
    create_splitter,
    get_next_file_num,
    add_config_to_argparser,
    make_evaluation_predictions_with_scaling,
    filter_metrics,
)
from uncond_ts_diff.model import LinearEstimator
from uncond_ts_diff.dataset import get_gts_dataset
import uncond_ts_diff.configs as diffusion_configs

#DOWNSTREAM_MODELS = ["linear", "deepar", "transformer"]
DOWNSTREAM_MODELS = ["transformer"]


def load_model(config):
    scheduler = config['schedule']
    if config['schedule']=='linear':
        if config['time_embed']:
            from uncond_ts_diff.model import TSDiff
        else:
            from uncond_ts_diff.model import TSDiff_wo_time as TSDiff
    else:
        assert config['time_embed']==0
        if config['schedule']=='sigmoid':
            from uncond_ts_diff.model import TSDiff_sigmoid as TSDiff
        elif config['schedule']=='cosine':
            from uncond_ts_diff.model import TSDiff_cosine as TSDiff
        else:
            import sys
            print("wrong scheduler!")
            sys.exit(0)
    
    a = getattr(diffusion_configs, "diffusion_small_config")
    a['timesteps'] = config['timesteps']

    if scheduler=='linear':
        model = TSDiff(
            **a,
            freq=config["freq"],
            use_features=config["use_features"],
            use_lags=config["use_lags"],
            normalization="mean",
            context_length=config["context_length"],
            prediction_length=config["prediction_length"],
            init_skip=config["init_skip"],
            device=config['device']
        )
    else:
        model = TSDiff(
            **a,
            freq=config["freq"],
            use_features=config["use_features"],
            use_lags=config["use_lags"],
            normalization="mean",
            context_length=config["context_length"],
            prediction_length=config["prediction_length"],
            init_skip=config["init_skip"],
            tau=config["tau"],
            device=config['device']
        )
        
    try:
        model.load_state_dict(
            torch.load(config["ckpt"], map_location="cpu"),
            strict=True,
        )
    except:
        model.load_state_dict(
            torch.load(config["ckpt"], map_location="cpu")['state_dict'],
            strict=True,
        )
        
    model = model.to(config["device"])
    return model


def sample_synthetic(
    model,
    num_samples: int = 10_000,
    batch_size: int = 1000,
):
    synth_samples = []

    n_iters = math.ceil(num_samples / batch_size)
    for _ in tqdm(range(n_iters)):
        samples = model.sample_n(num_samples=batch_size)
        synth_samples.append(samples)

    synth_samples = np.concatenate(synth_samples, axis=0)[:num_samples]

    return synth_samples


def sample_real(
    data_loader,
    n_timesteps: int,
    num_samples: int = 10_000,
    batch_size: int = 1000,
):
    real_samples = []
    data_iter = iter(data_loader)
    n_iters = math.ceil(num_samples / batch_size)
    for _ in tqdm(range(n_iters)):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)
        ts = np.concatenate(
            [batch["past_target"], batch["future_target"]], axis=-1
        )[:, -n_timesteps:]
        real_samples.append(ts)

    real_samples = np.concatenate(real_samples, axis=0)[:num_samples]

    return real_samples


def evaluate_tstr(
    tstr_predictor,
    test_dataset,
    context_length,
    prediction_length,
    num_samples=100,
    scaling_type="mean",
):
    total_length = context_length + prediction_length
    # Slice test set to be of the same length as context_length + prediction_length
    slice_func = partial(slice_data_entry, slice_=slice(-total_length, None))
    if scaling_type == "mean":
        ScaleAndAddScaleFeature = ScaleAndAddMeanFeature
    elif scaling_type == "min-max":
        ScaleAndAddScaleFeature = ScaleAndAddMinMaxFeature
    transformation = Chain(
        [
            AdhocTransform(slice_func),
            # Add scale to data entry for use later during evaluation
            ScaleAndAddScaleFeature("target", "scale", prediction_length),
        ]
    )
    sliced_test_set = transformation.apply(test_dataset)

    fcst_iter, ts_iter = make_evaluation_predictions_with_scaling(
        dataset=sliced_test_set,
        predictor=tstr_predictor,
        num_samples=num_samples,
        scaling_type=scaling_type,
    )
    evaluator = Evaluator()
    metrics, _ = evaluator(list(ts_iter), list(fcst_iter))
    return filter_metrics(metrics)


def train_and_evaluate(
    dataset,
    model_name,
    synth_samples,
    real_samples,
    config,
    scaling_type="mean",
):
    # NOTE: There's no notion of time for synthetic time series,
    # they are just "sequences".
    # A dummy timestamp is used for start time in synthetic time series.
    # Hence, time_features are set to [] in the models below.
    model_name = model_name.lower()
    freq = dataset.metadata.freq
    context_length = config["context_length"]
    prediction_length = config["prediction_length"]
    total_length = context_length + prediction_length

    assert len(synth_samples) == len(real_samples)
    assert (
        synth_samples.shape[-1] == total_length
        and real_samples.shape[-1] == total_length
    )
    num_samples = len(real_samples)

    synthetic_dataset = GluonTSNumpyDataset(synth_samples)

    if model_name == "linear":
        logger.info(f"Running TSTR for {model_name}")
        tstr_predictor = LinearEstimator(
            freq=freq,  # Not actually used in the estimator
            prediction_length=prediction_length,
            context_length=context_length,
            num_train_samples=num_samples,
            # Synthetic dataset is in the "scaled space"
            scaling=False,
        )#.train(synthetic_dataset)
    elif model_name == "deepar":
        logger.info(f"Running TSTR for {model_name}")
        tstr_predictor = DeepAREstimator(
            freq=freq,
            prediction_length=prediction_length,
            # Synthetic dataset is in the "scaled space"
            scaling=False,
            time_features=[],
            #learning_rate=0.001 *config['lr_scale'],
            lags_seq=get_lags_for_frequency(freq, lag_ub=context_length),
        )#.train(synthetic_dataset)
    elif model_name == "transformer":
        logger.info(f"Running TSTR for {model_name}")
        tstr_predictor = TransformerEstimator(
            freq=freq,
            prediction_length=prediction_length,
            # Synthetic dataset is in the "scaled space"
            scaling=False,
            time_features=[],
            lags_seq=get_lags_for_frequency(freq, lag_ub=context_length),
        )#.train(synthetic_dataset)
    tstr_predictor.trainer.learning_rate = tstr_predictor.trainer.learning_rate * config['lr_scale']
    tstr_predictor.trainer.epochs = tstr_predictor.trainer.epochs * config['epoch_alpha']
    tstr_predictor.trainer.num_batches_per_epoch = tstr_predictor.trainer.num_batches_per_epoch * config['epoch_alpha']
    tstr_predictor = tstr_predictor.train(synthetic_dataset)
       
    tstr_metrics = evaluate_tstr(
        tstr_predictor=tstr_predictor,
        test_dataset=dataset.test,
        context_length=context_length,
        prediction_length=prediction_length,
        scaling_type=scaling_type,
    )

    return dict(
        tstr_metrics=tstr_metrics,
    )


def main(config: dict, log_dir: str, samples_path: str):
    random_seed = config['seed']
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)    
        
    # Read global parameters
    dataset_name = config["dataset"]
    context_length = config["context_length"]
    prediction_length = config["prediction_length"]

    # Create log_dir
    log_dir: Path = Path(log_dir)
    #base_dirname = "tstr_log"
    #base_dirname = config['dataset_name']
    base_dirname = dataset_name
    #run_num = get_next_file_num(
    #    base_dirname, log_dir, file_type="", separator="-"
    #)
    #log_dir = log_dir / f"{base_dirname}-{run_num}"
    #log_dir.mkdir(exist_ok=True, parents=True)
    #logger.info(f"Logging to {log_dir}")
    #log_dir = log_dir / f"{base_dirname}" / config['setting']
    log_dir = log_dir / f"{base_dirname}" #/ config['setting']
    if os.path.exists(log_dir):
        import sys
        scale = args.ckpt.split('_')[-1].split('.ckpt')[0]
        if config['lr_scale']==1:
            PATH = f'results_{scale}_transformer_seed{random_seed}.yaml'
        else:
            PATH = f'results_{scale}_transformer_seed{random_seed}_lrs{config["lr_scale"]}.yaml'         
        if config['mm']:
            PATH = PATH.replace('.yaml','_mm.yaml')
        PATH = PATH.replace('.yaml',f"_ea{config['epoch_alpha']}_da{config['dataset_alpha']}.yaml")
        if os.path.isfile(log_dir / PATH):
            import sys
            print('already exists 111')
            sys.exit(0)
    else:
        log_dir.mkdir(exist_ok=True, parents=True)
        
    # Load dataset and model
    logger.info("Loading model")
    dataset = get_gts_dataset(dataset_name)
    config["freq"] = dataset.metadata.freq
    assert prediction_length == dataset.metadata.prediction_length

    model = load_model(config)

    # Setup data transformation and loading
    transformation = create_transforms(
        num_feat_dynamic_real=0,
        num_feat_static_cat=0,
        num_feat_static_real=0,
        time_features=time_features_from_frequency_str(config["freq"]),
        prediction_length=prediction_length,
    )
    transformed_data = transformation.apply(list(dataset.train), is_train=True)
    training_splitter = create_splitter(
        past_length=context_length + max(model.lags_seq),
        future_length=prediction_length,
        mode="train",
    )
    train_dataloader = TrainDataLoader(
        Cached(transformed_data),
        batch_size=1000,
        stack_fn=batchify,
        transform=training_splitter,
    )

    # Generate real samples
    logger.info("Generating real samples")
    fname = log_dir / "real_samples.npy"
    temp = 10000 * config['dataset_alpha']
    real_samples = sample_real(
            train_dataloader,
            n_timesteps=context_length + prediction_length,
            num_samples=temp,
        )
    np.save(fname, real_samples)
    '''
    if not os.path.isfile(fname):
        real_samples = sample_real(
            train_dataloader,
            n_timesteps=context_length + prediction_length,
            num_samples=temp,
        )
        np.save(fname, real_samples)
    else:
        print('real sample exists')
        real_samples = np.load(fname)[:temp]
        #print(real_samples.shape)[:temp]
        #[:temp]
    '''
    
    kk = log_dir / f"synth_samples_{scale}_seed{random_seed}_da{config['dataset_alpha']}.npy"
    
    if os.path.isfile(kk):
        logger.info(f"Using synthetic samples from {samples_path}")
        synth_samples = np.load(kk)[:temp]
        synth_samples = synth_samples.reshape(
            (temp, context_length + prediction_length)
        )
    else:
        logger.info("Generating synthetic samples")
        synth_samples = sample_synthetic(model, num_samples=temp)
        np.save(kk, synth_samples)
    '''
    if samples_path is None:
        # Generate synthetic samples
        logger.info("Generating synthetic samples")
        synth_samples = sample_synthetic(model, num_samples=temp)
        np.save(log_dir / f"synth_samples_last_seed{random_seed}_da{config['dataset_alpha']}.npy", synth_samples)
    else:
        try:
            logger.info(f"Using synthetic samples from {samples_path}")
            synth_samples = np.load(samples_path)[:temp]
            synth_samples = synth_samples.reshape(
                (temp, context_length + prediction_length)
            )
        except:
            logger.info("Generating synthetic samples")
            synth_samples = sample_synthetic(model, num_samples=temp)
            np.save(log_dir / f"synth_samples_last_seed{random_seed}_da{config['dataset_alpha']}.npy", synth_samples)
    '''
    # Run TSTR experiment for each downstream model
    results = []

    for model_name in DOWNSTREAM_MODELS:
        logger.info(f"Training and evaluating {model_name}")
        metrics = train_and_evaluate(
            dataset=dataset,
            model_name=model_name,
            synth_samples=synth_samples,
            real_samples=real_samples,
            config=config,
            scaling_type=config["scaling_type"],
        )
        results.append({"model": model_name, **metrics})

    logger.info("Saving results")
    scale = args.ckpt.split('_')[-1].split('.ckpt')[0]
    if config['lr_scale']==1:
        PATH = f"results_{scale}_transformer_seed{random_seed}.yaml"
    else:
        PATH = f"results_{scale}_transformer_seed{random_seed}_lrs{config['lr_scale']}.yaml"
    if config['mm']:
        PATH = PATH.replace('.yaml','_mm.yaml')
        
    PATH = PATH.replace('.yaml',f"_ea{config['epoch_alpha']}_da{config['dataset_alpha']}.yaml")
    with open(log_dir / PATH, "w") as fp:
        yaml.safe_dump(
            {"config": config, "metrics": results},
            fp,
            default_flow_style=False,
            sort_keys=False,
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
        "--out_dir", type=str, default="./tstr_results", help="Path to results dir"
    )
    parser.add_argument(
        "--samples_path", type=str, help="Path to generated samples"
    )
    parser.add_argument(
        "--time_embed", type=int, default=0
    )
    parser.add_argument(
        "--timesteps", type=int, default=100
    )
    parser.add_argument(
        "--schedule", type=str, default='linear',
    )
    parser.add_argument(
        "--tau", type=float, default=1
    )
    parser.add_argument(
        "--device", type=str, default='cuda:0',
    )
    parser.add_argument(
        "--ckpt", type=str, default='xx',
    )
    parser.add_argument(
        "--lr_scale", type=float, default=1
    )
    parser.add_argument(
        "--epoch_alpha", type=int, default=1,
    )
    parser.add_argument(
        "--dataset_alpha", type=int, default=1,
    )
    parser.add_argument(
        "--seed", type=int, default=1
    )    
    parser.add_argument(
        "--mm", type=float, default=0
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
    config["device"] = args.device
    config["timesteps"] = args.timesteps
    config["time_embed"] = args.time_embed
    config["schedule"] = args.schedule
    config["tau"] = args.tau
    config['seed'] = args.seed
    config['lr_scale'] = args.lr_scale
    config['mm'] = args.mm
    config['epoch_alpha'] = args.epoch_alpha
    config['dataset_alpha'] = args.dataset_alpha
    if args.mm:
        config['scaling_type'] = 'min-max'
    ######-----
    args.out_dir = args.out_dir.replace('tstr_results/',f'tstr_results/{config["dataset"]}/')
    ######-----
    
    if args.schedule=='linear':
        if args.time_embed==0:
            args.out_dir = args.out_dir.replace('results','results_wo_time')
        args.out_dir = args.out_dir.replace('results',f"results_T{args.timesteps}")
        # _s{args.ckpt.split('_')[-1].split('.ckpt')[0]
    else:
        assert args.time_embed==0
        args.out_dir = args.out_dir.replace('results','results_wo_time')
        args.out_dir = args.out_dir.replace('results',f"results_{args.schedule}_T{args.timesteps}_tau{args.tau}")
    
    print(args.out_dir)
    main(config=config, log_dir=args.out_dir, samples_path=args.samples_path)
