import comet_ml

import logging
from pathlib import Path

import hydra
import psutil
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from . import models
from .config import Config
from .data import cache_data, make_dataset
from .utils import setup_comet
from comet_ml.integration.pytorch import log_model


log = logging.getLogger(__name__)


Path.cwd()

@hydra.main(config_path=str(Path(__file__).parents[1] / "configs"), config_name="main", version_base="1.3")
def main(cfg: Config) -> None:
    comet_logger = setup_comet(cfg.comet.api_key, cfg.comet.project_name, cfg.comet.workspace)

    cache_data(cfg.input, cfg.paths.cache)
    train_dataset, test_dataset = make_dataset(
        cfg.input, cfg.splitting, cfg.preprocessing, split=True, cache_dir=cfg.paths.cache, read_ID_file=cfg.read_ID_file, only_aligned=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.n_data_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.n_data_workers
    )

    num_classes = len(train_loader.dataset.labels)
    model = models.SquiggleNet(models.Bottleneck, [2, 2, 2, 2], num_classes=num_classes, attribute=False)
    model.labels = train_loader.dataset.labels
    comet_logger.log_hyperparams(OmegaConf.to_container(cfg))
    comet_logger.log_hyperparams(
        {"num_classes": num_classes, "labels": train_loader.dataset.labels, "cmdline": " ".join(psutil.Process().cmdline()),
         "output_id": f"{Path.cwd().parent.name}/{Path.cwd().name}",
         },
    )

    trainer = pl.Trainer(
        devices=1,
        accelerator=cfg.training.accelerator,
        logger=comet_logger,
        max_epochs=cfg.training.max_epochs,
    )
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")
    trainer.fit(model, train_loader, test_loader)

    log_model(comet_logger.experiment, model, model_name=f"{comet_logger.experiment.name}_inference")

    comet_logger.experiment.end()


if __name__ == "__main__":
    main()
