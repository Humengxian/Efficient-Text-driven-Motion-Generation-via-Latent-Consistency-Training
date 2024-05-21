from .mae_trainer import MAETrainer
from .mlct_trainer import MLCTTrainer
from .mlct_tester import MLCTTester

def get_trainer(cfg, logger):
    if cfg.state == 'mae':
        return MAETrainer(cfg, logger)
    elif cfg.state == 'mlct':
        return MLCTTrainer(cfg, logger)
    
def get_tester(cfg, logger):
    return MLCTTester(cfg, logger)