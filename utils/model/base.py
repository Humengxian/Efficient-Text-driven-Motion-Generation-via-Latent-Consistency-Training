from .network.mae_net import TransformerMotionAutoEncoder
from .network.denoiser_net import TransformerMotionDenoiser

def get_models(cfg, model_type):
    if model_type == 'mae':
        return TransformerMotionAutoEncoder(cfg)
    elif model_type == 'denoiser':
        return TransformerMotionDenoiser(cfg)