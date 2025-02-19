import comet_ml

import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from . import models
from .config import Comet, Config, Splitting
from .data import cache_data, make_dataset
from .utils import get_comet_api_key, setup_comet

logger = logging.getLogger(__name__)


def get_cli_params_from_model(comet: Comet, experiment_id: str) -> Config:
    api = comet_ml.API(api_key=get_comet_api_key(comet.api_key))
    experiment = api.get(f"{comet.workspace}/{comet.project_name}/{experiment_id}")
    if experiment is None:
        raise ValueError(f"Experiment {experiment_id} not found")
    hparams = experiment.get_parameters_summary()

    def map2current(hparam):
        name = hparam["name"].replace("/", ".")
        return f"{name}={hparam['valueCurrent']}"

    def config_filter(hparam):
        return any([hparam["name"].startswith(prefix) for prefix in ["preprocessing", "comet"]])

    def extra_filter(hparam):
        return any([hparam["name"].startswith(prefix) for prefix in ["labels", "num_classes", "cmdline", "output_id"]])

    hparams_list = list(map(map2current, filter(config_filter, hparams)))
    if len(hparams) > 0:
        hdot = OmegaConf.create({"preprocessing": OmegaConf.to_container(hydra.compose("main", hparams_list, True))["preprocessing"]})
    else:
        hdot = OmegaConf.create({})

    extra_list = list(map(map2current, filter(extra_filter, hparams)))
    return hdot, OmegaConf.from_dotlist(extra_list)


@hydra.main(config_path=str(Path(__file__).parents[1] / "configs"), config_name="main", version_base="1.3")
def main(cfg: Config):
    if cfg.e is None:
        logger.error(
            "No checkpoint specified, please specify a checkpoint to load with the e=??? flag. Go on a comet.ml and find the experiment you want to load, then copy the experiment id (the long string of letters and numbers at the end of the URL) and pass it to the e flag. You can also run multiple by using the --multirun flag and give a list with comma."
        )
        exit(1)

    comet_logger = setup_comet(cfg.comet.api_key, cfg.comet.project_name, cfg.comet.workspace, cfg.e)
    hdot, extra_cfg = get_cli_params_from_model(cfg.comet, cfg.e)
    cfg = OmegaConf.merge(cfg, hdot)

    cache_data(cfg.input, cfg.paths.cache)
    dataset = make_dataset(cfg.input, Splitting(), cfg.preprocessing, split=False, cache_dir=cfg.paths.cache, read_ID_file=cfg.read_ID_file, only_aligned=cfg.only_aligned, comet_logger=comet_logger, test_data_name=cfg.test_data_name)
    d_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.n_data_workers
    )

    available_checkpoints = list(Path("checkpoints").glob("*.ckpt"))
    newest_checkpoint = max(available_checkpoints, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading checkpoint {newest_checkpoint}")

    model = models.SquiggleNet(models.Bottleneck, [2, 2, 2, 2], num_classes=extra_cfg.num_classes, attribute=False)
    model.labels = extra_cfg.labels
    qscore = cfg.input.min_qscore
    if cfg.test_data_name is not None:
        model.test_data_name = cfg.test_data_name
    else:
        model.test_data_name = f"q={qscore}_" + "_".join(inputset.name for inputset in cfg.input.inputset.values())
    model.curr_step = int(newest_checkpoint.stem.split("=")[-1])

    trainer = pl.Trainer(
        devices=1,
        accelerator=cfg.training.accelerator,
        logger=comet_logger,
    )
    trainer.test(model, d_loader, ckpt_path=newest_checkpoint)


if __name__ == "__main__":
    main()
