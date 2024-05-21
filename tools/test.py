from utils.model.base import get_models
from utils.config import test_parse_args
from utils.data.base import get_datasets
from utils.trainer.base import get_tester
import torch

def load_state_dict(path, model):
    checkpoint = torch.load(path, map_location='cpu')
    model_dict =  model.state_dict()
    state_dict = {k:v for k, v in checkpoint.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model

def main():
    # cfg & logger
    cfg, logger = test_parse_args()

    # create datasets
    datamodule = get_datasets(cfg)

    # create models
    mae = get_models(cfg, 'mae')
    mae = load_state_dict(cfg.motion_ae.pretrain_model_dir, mae)
    for p in mae.parameters():
        p.requires_grad = False

    denoiser = get_models(cfg, 'denoiser')
    denoiser = load_state_dict(cfg.diffusion.pretrain_model_dir, denoiser)
    for p in denoiser.parameters():
        p.requires_grad = False

    # create trainer
    Trainer = get_tester(cfg, logger)
    Trainer.test({'mae': mae, 'denoiser': denoiser}, datamodule)
    

if __name__ == "__main__":
    main()