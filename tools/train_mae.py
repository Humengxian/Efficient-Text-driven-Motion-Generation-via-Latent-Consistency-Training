from utils.config import train_mae_parse_args
from utils.data.base import get_datasets
from utils.model.base import get_models
from utils.trainer.base import get_trainer

def main():
    # cfg & logger
    cfg, logger = train_mae_parse_args()

    # create datasets
    datamodule = get_datasets(cfg)

    # create models
    mae = get_models(cfg, 'mae')

    # create trainer
    Trainer = get_trainer(cfg, logger)
    Trainer.train({'mae': mae}, datamodule)
    

if __name__ == "__main__":
    main()