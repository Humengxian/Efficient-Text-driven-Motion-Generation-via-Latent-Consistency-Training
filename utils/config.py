import argparse
import time
import os
from omegaconf import OmegaConf
import logging
import json
import torch
import numpy as np
import random

def create_logger(cfg):
    # make logger
    os.makedirs(cfg.output_dir, exist_ok=True)
    logger = logger_maker(cfg)

    # save and report hyper parameters
    logger.info(f'[INFO] Report hyper parameters:')
    for key, value in cfg.items():
        if type(value) not in [str, int, float, bool]:
            for subkey, subvalue in value.items():
                logger.info(f'{key}.{subkey}: {subvalue}')
        else:
            logger.info(f'{key}: {value}')
    return logger

def logger_maker(cfg):
    log_file = f'running_log.log'
    final_log_file = os.path.join(cfg.output_dir, log_file)
    logging.basicConfig(filename=final_log_file)
    head = '%(asctime)-15s %(message)s'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    file_handler = logging.FileHandler(final_log_file, 'w')
    file_handler.setFormatter(logging.Formatter(head))
    file_handler.setLevel(logging.INFO)
    logging.getLogger('').addHandler(file_handler)
    return logger

def init(params):
    # update config from files
    cfg = OmegaConf.load(params.cfg)
    cfg.name = params.cfg.split('/')[-1].split('.')[0]
    cfg.seed = params.seed
    cfg.device = params.device
    cfg.gpu = params.gpu
    params.exp = f'_{params.exp}' if len(params.exp) != 0 else params.exp
    cfg.output_dir = os.path.join('experiments', cfg.name, time.strftime('%Y-%m-%d-%H-%M-%S') + params.exp)
    cfg.save_dir = os.path.join(cfg.output_dir, 'checkpoints')
    cfg.hyperpara_dir = os.path.join(cfg.output_dir, 'hyperparams')
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.hyperpara_dir, exist_ok=True)

    return cfg

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_mae_parse_args():
    # argument
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("Options")
    group.add_argument("-e", "--exp", type=str, required=False, default="", help="experiment name")
    group.add_argument("-c", "--cfg", type=str, required=False, default="configs/humanml/ablation/humanml_mae_qae_tn_2.yaml", help="config file")
    group.add_argument("-s", "--seed", type=int, required=False, default=42, help="random seed")
    group.add_argument("-d", "--device", type=str, required=False, default='cuda:', help="device")
    group.add_argument("-g", "--gpu", type=int, required=False, default=0, help="device")
    params = parser.parse_args()
    cfg = init(params)

    # logger
    logger = create_logger(cfg)

    # seed
    seed_everything(cfg.seed)

    with open(os.path.join(cfg.hyperpara_dir, 'motion_ae_hparams.json'), 'w') as fw:
        json.dump({'motion_ae': dict(cfg.motion_ae)}, fw, sort_keys=True)
        
    return cfg, logger

def train_mlct_parse_args():
    # argument
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("Options")
    group.add_argument("-e", "--exp", type=str, required=False, default="", help="experiment name")
    group.add_argument("-c", "--cfg", type=str, required=False, default="configs/kit/ablation/kit__w_1.yaml", help="config file")
    group.add_argument("-s", "--seed", type=int, required=False, default=42, help="random seed")
    group.add_argument("-d", "--device", type=str, required=False, default='cuda:', help="device")
    group.add_argument("-g", "--gpu", type=str, required=False, default='0', help="device")
    params = parser.parse_args()
    cfg = init(params)

    # seed
    seed_everything(cfg.seed)

    # merge motion ae net parameters
    with open(os.path.join(cfg.motion_ae.pretrain_dir, 'hyperparams', 'motion_ae_hparams.json'), 'r') as f:
        cfg.merge_with(json.load(f))

    with open(os.path.join(cfg.hyperpara_dir, 'diffusion_hparams.json'), 'w') as fw:
        json.dump({'diffusion': dict(cfg.diffusion)}, fw, sort_keys=True)

    if cfg.train.visual_epochs != 0:
        cfg.sample_dir = os.path.join(cfg.output_dir, 'sample')
    
    os.makedirs(cfg.sample_dir, exist_ok=True)
    cfg.motion_ae.pretrain_model_dir = os.path.join(cfg.motion_ae.pretrain_dir, 'checkpoints', cfg.motion_ae.pretrain_model_name)
    
    # logger
    logger = create_logger(cfg)

    # gpu setting
    if cfg.accelerator == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    return cfg, logger
 
def test_parse_args():
    # argument
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("Options")
    group.add_argument("-e", "--exp", type=str, required=False, default="", help="experiment name")
    group.add_argument("-c", "--cfg", type=str, required=False, default="configs/kit/ablation/kit__w_1.yaml", help="config file")
    group.add_argument("-s", "--seed", type=int, required=False, default=42, help="random seed")
    group.add_argument("-d", "--device", type=str, required=False, default='cuda:', help="device")
    group.add_argument("-g", "--gpu", type=str, required=False, default='0', help="device")
    params = parser.parse_args()
    # update config from files
    cfg = OmegaConf.load(params.cfg)
    cfg.name = params.cfg.split('/')[-1].split('.')[0]
    cfg.seed = params.seed
    cfg.device = params.device
    cfg.gpu = params.gpu
    params.exp = f'_{params.exp}' if len(params.exp) != 0 else params.exp
    cfg.output_dir = os.path.join('experiments', cfg.name, time.strftime('%Y-%m-%d-%H-%M-%S') + params.exp)
    cfg.save_dir = os.path.join(cfg.output_dir, 'checkpoints')
    cfg.hyperpara_dir = os.path.join(cfg.output_dir, 'hyperparams')

    # seed
    seed_everything(cfg.seed)

    # merge motion ae net parameters
    with open(os.path.join(cfg.motion_ae.pretrain_dir, 'hyperparams', 'motion_ae_hparams.json'), 'r') as f:
        cfg.merge_with(json.load(f))

    # merge motion ae net parameters
    with open(os.path.join(cfg.diffusion.pretrain_dir, 'hyperparams', 'diffusion_hparams.json'), 'r') as f:
        cfg.merge_with(json.load(f))

    cfg.motion_ae.pretrain_model_dir = os.path.join(cfg.motion_ae.pretrain_dir, 'checkpoints', cfg.motion_ae.pretrain_model_name)
    cfg.diffusion.pretrain_model_dir = os.path.join(cfg.diffusion.pretrain_dir, 'checkpoints', cfg.diffusion.pretrain_model_name)

    # logger
    logger = create_logger(cfg)

    # gpu setting
    if cfg.accelerator == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    return cfg, logger

def sample_parse_args():
    # argument
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("Options")
    group.add_argument("-e", "--exp", type=str, required=False, default="Dubug", help="experiment name")
    group.add_argument("-bs", "--bs", type=int, required=False, default=1, help="batch size")
    group.add_argument("-c", "--cfg", type=str, required=False, default="configs/humanml/humanml_sample.yaml", help="config file")
    group.add_argument("-r", "--src", type=str, required=False, default="a descends into a falling motion and thens bounces back up.", help="input srouce")
    group.add_argument("-s", "--seed", type=int, required=False, default=42, help="random seed")
    group.add_argument("-d", "--device", type=str, required=False, default='cuda:', help="device")
    group.add_argument("-g", "--gpu", type=str, required=False, default='0', help="device")
    params = parser.parse_args()

    # update config from files
    cfg = OmegaConf.load(params.cfg)
    
    cfg.name = params.cfg.split('/')[-1].split('.')[0]
    params.exp = f'_{params.exp}' if len(params.exp) != 0 else params.exp
    cfg.seed = params.seed
    cfg.device = params.device
    cfg.bs = params.bs
    cfg.gpu = params.gpu
    cfg.output_dir = os.path.join('sample', cfg.name, time.strftime('%Y-%m-%d-%H-%M-%S') + params.exp)
    cfg.sample_dir = os.path.join(cfg.output_dir, 'sample')
    os.makedirs(cfg.sample_dir, exist_ok=True)

    # seed
    seed_everything(cfg.seed)

    # merge motion ae net parameters
    with open(os.path.join(cfg.motion_ae.pretrain_dir, 'hyperparams', 'motion_ae_hparams.json'), 'r') as f:
        cfg.merge_with(json.load(f))

    with open(os.path.join(cfg.diffusion.pretrain_dir, 'hyperparams', 'diffusion_hparams.json'), 'r') as f:
        cfg.merge_with(json.load(f))

    cfg.motion_ae.pretrain_model_dir = os.path.join(cfg.motion_ae.pretrain_dir, 'checkpoints', cfg.motion_ae.pretrain_model_name)
    cfg.diffusion.pretrain_model_dir = os.path.join(cfg.diffusion.pretrain_dir, 'checkpoints', cfg.diffusion.pretrain_model_name)

    # srouce
    src = params.src
    if src[-3:] == 'txt':
        pass
    else:
        cfg.src = src

    # logger
    logger = create_logger(cfg)

    # gpu setting
    if cfg.accelerator == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    return cfg, logger