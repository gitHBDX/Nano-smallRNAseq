import logging
import subprocess
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import comet_ml
import yaml
from rich.logging import RichHandler

# CONFIG .......................................................................
REFERENCES = {}

BARCODES = {"1to3", "1to6"}

EXPERIMENTS = {
    "barcoding": {
        "1to3": [],
        "1to6": [],
    },
    "methylation": {
        "milung1": [],
        "mir17": [],
    },
}

MACHINE_FOLDERS = [
    "/",
    "/media/illumina/MinION_mk1b_MN_45231/",
    "/media/illumina/MinION_mk1c_MC_115192/",
    "/media/illumina/MinION_mk1b_MN_45231",
]
# ..............................................................................

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
log = logging.getLogger("ont_hbdx.auto")
GLOBAL_ERRORS = []

comet_api = comet_ml.API(api_key=Path("./comet_api_key.txt").read_text())


def execute_prediction(
    inputset: str,
    model_id: str,
    report: dict,
    process_config: dict,
    overwrite: bool = False,
    dry_run: bool = False,
):
    log.info(f"Running {inputset} model {model_id} [process_config={process_config}, overwrite={overwrite}]")
    if not (experiment := comet_api.get(f"morris-frank/nanopore/{model_id}")):
        log.error(f"Experiment {model_id} not found")
        GLOBAL_ERRORS.append(f"Experiment {model_id} not found")
        return report

    predictions_path = Path(f"nanopore/{experiment.id}/inference_{inputset}.csv").absolute()
    if not overwrite and predictions_path.exists():
        log.info(f"Skipping {experiment.name}, already predicted")
        return report

    cmd = [
        "python",
        "-m",
        "ont_hbdx.test",
        f"+input/inputset=[{inputset}]",
        f"e={experiment.id}",
        f"input.min_qscore={process_config['qscore']}",
        f"input.kit={process_config['basecalling_model']}",
        f"test_data_name={inputset}",
        f"only_aligned={not process_config['include_unaligned']}",
    ]
    log.info(f"Running {cmd}")
    if not dry_run:
        subp = subprocess.run(cmd)
        if subp.returncode != 0:
            log.error(f"Failed to run {experiment.name} on {inputset}")
            GLOBAL_ERRORS.append(f"Failed to run {experiment.name} on {inputset}")
            return report

    cmdline = experiment.get_parameters_summary("cmdline")
    cmdline = "" if len(cmdline) == 0 else cmdline["valueCurrent"]
    report[experiment.name] = {
        "experiment_id": experiment.id,
        "name": experiment.name,
        "url": experiment.url,
        "inference": str(predictions_path),
        "start_server_timestamp": experiment.start_server_timestamp,
        "end_server_timestamp": experiment.end_server_timestamp,
        "cmdline": cmdline,
        "process_time": datetime.now().isoformat(),
        **process_config,
    }
    return report


def resolve_inpuset(path: str, reference: str, pod5: bool = False) -> str | None:
    for folder in MACHINE_FOLDERS:
        root_path = Path(folder) / path
        if not root_path.exists():
            continue

        folder_name = "pod5" if pod5 else "fast5"
        folders = [d for d in root_path.rglob(folder_name) if d.is_dir()]
        if len(folders) == 0:
            log.error(f"No fast5 folders found in {root_path}")
            return None
        if len(folders) > 1:
            log.error(f"Multiple fast5 folders found in {root_path}")
            return None

        fast5_folder = folders[0]
        inputset = fast5_folder.parts[4]

        Path(f"configs/input/inputset/{inputset}.yaml").write_text(
            f"""{inputset}:
            name: {inputset}
            label: unknown
            reference: {REFERENCES[reference]}
            fast5: {fast5_folder}/"""
        )

        log.info(f"Using config {inputset} for {path}.")
        return inputset

    else:
        log.error(f"Path {path} could not be localized.")
        return None


def auto_predict_path(
    name: str, reference: str, barcodes: str, process_config: dict, overwrite: bool = False, dry_run: bool = False, pod5: bool = False
):
    if not (inputset := resolve_inpuset(name, reference, pod5)):
        log.error(f"Skipping {name}")
        GLOBAL_ERRORS.append(f"Could not resolve inputset for {name}.")
        return

    if (report_path := Path(f"reports/{inputset}.yaml")).exists():
        report = yaml.load(report_path.read_text(), Loader=yaml.FullLoader)
        log.info(f"Loaded report from {report_path}")
    else:
        report = {}

    for target, labels in [("barcoding", barcodes), ("methylation", reference)]:
        if target not in report:
            report[target] = {}
        for model_id in EXPERIMENTS[target][labels]:
            report[target] = execute_prediction(inputset, model_id, report[target], process_config, overwrite, dry_run)
            if not dry_run:
                report_path.write_text(yaml.dump(report))


parser = ArgumentParser()
parser.add_argument("names", help="List of paths or names of inputs", nargs="+")
parser.add_argument("-f", "--overwrite", action="store_true", help="Overwrite existing results")
parser.add_argument("--dry-run", action="store_true", help="Do not run the predictions")
pp_parser = parser.add_argument_group("Preprocessing options")
pp_parser.add_argument("--qscore", type=str, default="-1", help="min q-score to test, default: 2 (for pod5 4)")
pp_parser.add_argument("--include_unaligned", action="store_true", help="Include unaligned reads, default: False")
pp_parser.add_argument("--barcodes", choices=["1to3", "1to6", "auto"], default="auto", help="Barcodes to use, default: auto pick")
pp_parser.add_argument("--pod5", action="store_true", help="Use pod5 reference")
args = parser.parse_args()

q_score = (10 if args.pod5 else 2) if args.qscore == "-1" else args.qscore
process_config = {
    "qscore": q_score,
    "basecalling_model": "SQK-RNA004" if args.pod5 else "SQK-RNA002",
    "include_unaligned": args.include_unaligned,
}

for name in args.names:
    if "mir21" in name.lower():
        print("Skipping mir21")
        continue
    reference = "mir17" if "mir17" in name.lower() else "milung1"
    if args.barcodes == "auto":
        barcodes = "1to6" if reference == "milung1" else "1to3"
    else:
        barcodes = args.barcodes
    auto_predict_path(name, reference, barcodes, process_config, args.overwrite, args.dry_run, args.pod5)

if len(GLOBAL_ERRORS) > 0:
    log.error("Errors occurred:")
    for error in GLOBAL_ERRORS:
        log.error(error)
