import os
import logging
from omegaconf import DictConfig

from .utils import (
    set_seed,
    get_dataloaders,
    initialize_logger,
    initialize_trainer,
    load_obj,
)

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(levelname)s:%(message)s", datefmt="%d:%m:%Y|%H:%M:%S")


def train_pipeline(cfg: DictConfig) -> None:
    os.makedirs(cfg.general.artifacts_dir, exist_ok=True)
    os.makedirs(cfg.general.checkpoint_path, exist_ok=True)
    set_seed(cfg.general.seed)
    logger = initialize_logger(cfg.logger)
    train_loader, valid_loader = get_dataloaders(cfg)
    trainer = initialize_trainer(cfg, logger)
    trainer.train(train_loader, valid_loader)


def test_pipeline(cfg: DictConfig) -> None:
    set_seed(cfg.general.seed)
    logger = initialize_logger(cfg.logger)
    test_loader = get_dataloaders(cfg, stages=["test"])
    trainer = initialize_trainer(cfg, logger)
    trainer.test(test_loader)
