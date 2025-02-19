import dataclasses
import functools
import logging
import os
import timeit
from pathlib import Path

import hydra
from pytorch_lightning.loggers import CometLogger


def humanize_time(t: float) -> str:
    if t < 1:
        return f"{t * 1000:>8.2f} ms"
    elif t < 60:
        return f"{t:>8.2f} s "
    else:
        return f"{t / 60:>8.2f} min"


@dataclasses.dataclass
class log_time:
    """Context manager and decorator to time code blocks.

    Usage:
    ```
    with log_time():
        # Code to time
    ```

    or

    ```
    @log_time()
    def func():
        # Code to time
    ```
    """

    name: str = None
    logger: logging.Logger = None
    prefix: str = None
    suffix: str = None

    def __enter__(self) -> None:
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        timedelta = humanize_time(timeit.default_timer() - self.start)
        prefix = self.prefix or "block   "
        suffix = self.suffix or ""
        message = f"{timedelta} for {prefix} '{self.name}' {suffix}"

        if self.logger is None:
            print(message)
        else:
            self.logger.info(message)

    def __call__(self, func):
        if self.name is None:
            self.name = func.__qualname__
        if self.prefix is None:
            self.prefix = "function"

        @functools.wraps(func)
        def inner(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return inner


def get_comet_api_key(comet_api_key_file: str):
    logger = logging.getLogger("comet_ml")
    comet_api_key_file = Path(hydra.utils.get_original_cwd()) / comet_api_key_file
    if not comet_api_key_file.exists():
        logger.error(
            f"Comet API key file does not exist, expected it at the location {comet_api_key_file}. Please create an empty file at this path and copy you Comet ML api key into it. Then restart."
        )
        exit(1)
    comet_api_key = comet_api_key_file.read_text().strip()
    logger.info(f"Comet API key: {comet_api_key[:4]}...{comet_api_key[-4:]}")
    return comet_api_key


def setup_comet(comet_api_key_file: str, project_name: str, workspace: str, experiment: str | None = None) -> CometLogger:
    comet_api_key = get_comet_api_key(comet_api_key_file)

    comet_logger = CometLogger(
        api_key=comet_api_key,
        project_name=project_name,
        workspace=workspace,
        experiment_key=experiment,
        save_dir=Path.cwd(),
    )

    working_dir = Path(project_name) / comet_logger.experiment.id
    working_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(working_dir)
    return comet_logger
