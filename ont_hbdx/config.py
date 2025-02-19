from pathlib import Path
from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class Inputset:
    name: str
    fast5: str
    label: str
    min_length: int = 2000
    id_file: str | None = None
    reference: Path | None = None


@dataclass
class Input:
    inputset: dict[str, Inputset]
    flowcell: str
    kit: str
    guppy_device: str
    min_qscore: int


@dataclass
class Splitting:
    test_size: float = 0.2
    n_per_label: int = 50_000
    random_state: int = 42


@dataclass
class Preprocessing:
    mad_normalize: bool = False
    scale_to_fixed_length: int = 0
    remove_outliers: bool = False
    remove_outliers_max_std: int = 3
    remove_outliers_window_size: int = 800
    left_padding: int = 0
    window_size: int = 0
    rupture: bool = False
    rupture_part: str = "RNA"


@dataclass
class Training:
    max_epochs: int = 100
    batch_size: int = 32
    n_data_workers: int = 4
    accelerator: str = "gpu"


@dataclass
class Comet:
    api_key: str
    project_name: str
    workspace: str


@dataclass
class Paths:
    cache: Path
    output: Path

@dataclass
class Config:
    input: Input
    splitting: Splitting
    preprocessing: Preprocessing
    training: Training
    comet: Comet
    paths: Paths
    e: str | None = None # Experiment ID
    test_data_name: str | None = None
    read_ID_file: str | None = None # Read ID file
    only_aligned: bool = False


cs = ConfigStore.instance()
# name `base_config` is used for matching it with the main.yaml's default section
cs.store(name="base_config", node=Config)
