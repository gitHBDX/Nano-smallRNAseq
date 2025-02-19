import hashlib
import pod5
from Bio import SeqIO
import logging
from pytorch_lightning.loggers import CometLogger
import multiprocessing
import subprocess
from pathlib import Path

import pandas as pd
import torch
import torchvision
from joblib import Parallel, delayed
import pysam
from ont_fast5_api.fast5_interface import get_fast5_file
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from . import transforms
from .config import Input, Preprocessing, Splitting
from .utils import log_time

log = logging.getLogger(__name__)


def copy_fast5_files(from_: Path, to_: Path) -> None:
    with log_time(name=f"rsync {from_} to {to_}", logger=log):
        subprocess.run(["rsync", "-avP", "--ignore-existing", str(from_).rstrip("/"), to_], check=True)


def guppy_basecaller(from_: Path, to_: Path, data: Input):
    if to_.exists():
        log.info(f"Skip guppy {from_} to {to_}, already exists")
        return

    with log_time(name=f"guppy {from_} to {to_}", logger=log):
        # subprocess.run(
        #     [
        #         "guppy_basecaller",
        #         "-i",
        #         str(from_),
        #         "-s",
        #         str(to_),
        #         "--flowcell",
        #         data.flowcell,
        #         "--kit",
        #         data.kit,
        #         "--disable_qscore_filtering",
        #         "--records_per_fastq",
        #         "0",
        #         "--device",
        #         data.guppy_device,
        #         "--moves_out",
        #         "--u_substitution",
        #         "--bam_out",
        #     ],
        #     check=True,
        # )
        subprocess.run(
            [
                "guppy_basecaller",
                "-i",
                str(from_),
                "-s",
                str(to_),
                "-c",
                "rna_r9.4.1_70bps_sup.cfg",
                "--num_callers",
                "2",
                "--cpu_threads_per_caller",
                "1",
                "--disable_qscore_filtering",
                "--records_per_fastq",
                "0",
                "--device",
                data.guppy_device,
                "--moves_out",
                "--u_substitution",
                "--bam_out",
            ]
        )


def dorado_basecaller(from_: Path, to_: Path, data: Input):
    # create single fastq file in the output directory
    fastq_path = to_ / "dorado.fastq"
    sequence_summary = to_ / "sequencing_summary.txt"
    bam_path = to_ / "dorado.bam"

    if fastq_path.exists():
        log.info(f"Skip dorado {from_} to {to_}, already exists")
        return

    to_.mkdir(parents=True, exist_ok=True)
    fastq_path.touch()
    sequence_summary.touch()
    bam_path.touch()

    stdout_fastq = fastq_path.open("w")
    stdout_sequence_summary = sequence_summary.open("w")
    stdout_bam = bam_path.open("w")
    with log_time(name=f"dorado {from_} to {to_}", logger=log):
        subprocess.run(
            [
                "dorado",
                "basecaller",
                "/home/mfr_local/code/nanopore/rna004_130bps_sup@v5.0.0",
                str(from_),
                "--no-trim",
                "--min-qscore",
                str(data.min_qscore),
            ],
            stdout=stdout_bam,
            check=True,
        )
    with log_time(name=f"dorado {from_} to {to_}", logger=log):
        subprocess.run(
            [
                "dorado",
                "basecaller",
                "/home/mfr_local/code/nanopore/rna004_130bps_sup@v5.0.0",
                str(from_),
                "--no-trim",
                "--emit-fastq",
                "--min-qscore",
                str(data.min_qscore),
            ],
            stdout=stdout_fastq,
            check=True,
        )
    with log_time(name=f"Indexing {fastq_path}", logger=log):
        subprocess.run(
            [
                "dorado",
                "summary",
                str(fastq_path),
            ],
            stdout=stdout_sequence_summary,
            check=True,
        )


def bwa_align(from_: Path, to_: Path, alignment_file: Path):
    to_.mkdir(parents=True, exist_ok=True)
    fastq_files = [f for f in from_.glob("*.fastq") if not f.name.endswith(".dna.fastq")]

    alignment_summary = to_ / "alignment_summary.txt"
    aligned_IDs = set()

    for fastq in tqdm(fastq_files, desc="Aligning"):
        dna_fastq = fastq.with_suffix(".dna.fastq")
        if not dna_fastq.exists():
            subprocess.run(f"seqkit seq --rna2dna {fastq} -o {dna_fastq}".split(" "), check=True)

        sam = to_ / fastq.name.replace(".fastq", ".sam")
        if not sam.exists():
            subprocess.run(f"bwa mem -W 13 -k 6 -xont2d -T 20 {alignment_file} {dna_fastq} -o {sam}".split(" "), check=True)

            current_input = sam
            for lvl, ref in enumerate(["", " Nanopore_reference:20-25", " Nanopore_reference:60-65"]):
                bam_file = sam.with_suffix(f".{lvl}.bam")
                bam_sorted_file = bam_file.with_suffix(".sorted.bam")

                subprocess.run(f"samtools view -b -o {bam_file} {current_input}{ref}".split(" "), check=True)
                # Explanation of the samtools flags:
                # view: convert to bam
                # -b: output in bam format
                # -o: output file

                subprocess.run(f"samtools sort -o {bam_sorted_file} {bam_file}".split(" "), check=True)
                # Explanation of the samtools flags:
                # sort: sort the bam file
                # -o: output file

                subprocess.run(f"samtools index {bam_sorted_file}".split(" "), check=True)
                # Explanation of the samtools flags:
                # index: create an index for the bam file

                current_input = bam_sorted_file

            samfile = pysam.AlignmentFile(current_input, "r")
            for i in samfile.fetch("Nanopore_reference"):
                aligned_IDs.add(str(i).split("\t")[0])

    if not alignment_summary.exists():
        alignment_summary.touch()
    if len(aligned_IDs) > 0:
        alignment_summary.write_text("\n".join(set(alignment_summary.read_text().splitlines()).union(aligned_IDs)))


def read_fast5_file(file: Path, seq_summary: pd.DataFrame, folder: Path, min_length: int = -1):
    PASS_reads = set(seq_summary["read_id"])
    with get_fast5_file(file, mode="r") as f5, log_time(name=f"read {file}", logger=log):
        for i, read in enumerate(f5.get_reads()):
            if read.read_id not in PASS_reads:
                continue
            signal = read.get_raw_data().squeeze()
            if min_length > 0 and len(signal) < min_length:
                continue
            try:
                torch.save(torch.from_numpy(signal), folder / f"{read.read_id}.pt")
            except RuntimeError as e:
                print(e)
                continue
    log.info(f"Read {i} reads from {file}")


def read_pod5_file(file: Path, seq_summary: pd.DataFrame, folder: Path, min_length: int = -1):
    PASS_reads = set(seq_summary["read_id"])
    with pod5.Reader(file) as reader:
        for i, read_record in enumerate(reader.reads(PASS_reads, missing_ok=True)):
            signal = read_record.signal.squeeze()
            # if min_length > 0 and len(signal) < min_length:
            #     continue
            try:
                torch.save(torch.from_numpy(signal), folder / f"{read_record.read_id}.pt")
            except RuntimeError as e:
                print(e)
                continue
    log.info(f"Read {i} reads from {file}")


def to_torch(function, from_: Path, to_: Path, min_length: int, summary_file: Path):
    from_ = Path(from_)
    if to_.exists():
        log.info(f"Skip torch conversion {from_} to {to_}, already exists")
        return

    to_.mkdir(parents=True, exist_ok=True)

    seq_summary = pd.read_csv(summary_file, sep="\t")

    log.info(f"Converting torched {from_} to {to_}")
    Parallel(n_jobs=int(multiprocessing.cpu_count() * 0.8))(
        delayed(function)(from_ / name, group, to_, min_length) for name, group in seq_summary.groupby("filename")
    )


def cache_data(input: Input, cache_dir: Path) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for inputset in input.inputset.values():
        inputset_dir = cache_dir / inputset.name
        inputset_dir.mkdir(parents=True, exist_ok=True)

        fast5_dir = inputset_dir / "fast5"
        torch_dir = inputset_dir / "torch"
        guppy_dir = inputset_dir / "guppy"
        align_dir = inputset_dir / "align"
        seq_summary_file = guppy_dir / "sequencing_summary.txt"

        # try:
        #     copy_fast5_files(inputset.fast5, inputset_dir)
        # except OSError:
        #     log.warning(f"Could not copy {inputset.fast5}, skipping")
        #     continue
        if input.kit == "SQK-RNA004":
            dorado_basecaller(inputset.fast5, guppy_dir, input)
        else:
            guppy_basecaller(inputset.fast5, guppy_dir, input)
        if inputset.reference is not None:
            bwa_align(guppy_dir, align_dir, Path(inputset.reference))
        if input.kit == "SQK-RNA004":
            to_torch(read_pod5_file, inputset.fast5, torch_dir, inputset.min_length, seq_summary_file)
        else:
            to_torch(read_fast5_file, inputset.fast5, torch_dir, inputset.min_length, seq_summary_file)


class ReadDataset(torch.utils.data.Dataset):
    def __init__(self, filesheet, labels: list[str], transform=None, comet_logger: CometLogger = None, name: str = None):
        self.filesheet = filesheet
        self.transform = transform
        self.labels = labels
        self.comet_logger = comet_logger

        self._data = []
        self._meta = []
        self.not_found = []
        for _, row in tqdm(self.filesheet.iterrows(), total=len(self.filesheet)):
            row = row.to_dict()
            path = Path(row.pop("path"))
            if path.exists():
                data = torch.load(path)
                self._data.append(self.transform(data))
                self._meta.append({"y": labels.index(row["label"]), **row})
            else:
                self.not_found.append(path)
        if comet_logger is not None:
            comet_logger.log_metrics({f"{name}/skipped-reads": len(self.not_found)})

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx], self._meta[idx]


def read_read_ID_file(path: Path | str) -> set[str]:
    path = Path(path)

    if not path.absolute():
        path = Path(__file__).parents[1] / path

    if not path.exists():
        raise FileNotFoundError(f"ReadID File {path} does not exist")

    with open(path, "r") as f:
        read_IDs = set(f.read().splitlines())
    log.info(f"Loaded {len(read_IDs)} read IDs from {path}")
    return read_IDs


def filter_read_IDs(seq_summary: pd.DataFrame, read_IDs: set[str]) -> pd.DataFrame:
    log.info(f"Filtering by read IDs, {len(read_IDs)} valid read IDs")
    before = len(seq_summary)
    seq_summary = seq_summary[seq_summary["read_id"].isin(read_IDs)]
    log.info(f"Before: {before}, after: {len(seq_summary)}")
    return seq_summary


def make_dataset(
    input: Input,
    splitting: Splitting,
    preprocessing: Preprocessing,
    cache_dir: Path,
    split: bool = True,
    read_ID_file: str | None = None,
    only_aligned: bool = False,
    comet_logger: CometLogger | None = None,
    test_data_name: str = "test",
):
    cache_dir = Path(cache_dir)

    m = hashlib.sha256()
    m.update(str(input).encode("utf-8"))
    m.update(str(splitting).encode("utf-8"))
    m.update(str(preprocessing).encode("utf-8"))
    m.update(str(read_ID_file).encode("utf-8"))
    hash = m.hexdigest()

    train_dataset_path = cache_dir / f"dataset_train_{hash}.pt"
    test_dataset_path = cache_dir / f"dataset_test_{hash}.pt"
    dataset_path = cache_dir / f"dataset_{hash}.pt"
    if split:
        print(train_dataset_path, "\n", test_dataset_path)
    else:
        print(dataset_path)

    global_valid_read_IDs = None
    if read_ID_file is not None:
        global_valid_read_IDs = read_read_ID_file(read_ID_file)

    if (split and (train_dataset_path.exists() and test_dataset_path.exists())) or (not split and dataset_path.exists()):
        if split:
            train_dataset = torch.load(train_dataset_path)
            test_dataset = torch.load(test_dataset_path)
            return train_dataset, test_dataset
        else:
            dataset = torch.load(dataset_path)
            return dataset
    else:
        with log_time("Creating file-sheet", logger=log):
            files = []
            lables = []
            qscores = []
            readIDs = []
            aligned = []
            label_order = []

            for inputset in input.inputset.values():
                inputset_dir = cache_dir / inputset.name
                torch_dir = inputset_dir / "torch"
                guppy_dir = inputset_dir / "guppy"
                align_dir = inputset_dir / "align"
                seq_summary_file = guppy_dir / "sequencing_summary.txt"
                seq_summary = pd.read_csv(seq_summary_file, sep="\t")
                if input.kit != "SQK-RNA004":
                    seq_summary = seq_summary[seq_summary["mean_qscore_template"] >= input.min_qscore]

                align_summary = align_dir / "alignment_summary.txt"
                aligned_IDs = set(align_summary.read_text().splitlines())
                aligned.extend([read_id in aligned_IDs for read_id in seq_summary["read_id"]])

                if global_valid_read_IDs is not None:
                    seq_summary = filter_read_IDs(seq_summary, global_valid_read_IDs)

                if hasattr(inputset, "id_file") and inputset.id_file is not None:
                    valid_read_IDs = read_read_ID_file(inputset.id_file)
                    seq_summary = filter_read_IDs(seq_summary, valid_read_IDs)

                _files = [torch_dir / (read_id + ".pt") for read_id in seq_summary["read_id"]]
                files.extend(_files)
                lables.extend([inputset.label] * len(_files))
                qscores.extend(seq_summary["mean_qscore_template"].tolist())
                readIDs.extend(seq_summary["read_id"].tolist())
                if inputset.label not in label_order:
                    label_order.append(inputset.label)
                log.info(f"Found {len(_files)} files in {inputset.name}, label: {inputset.label}")
            filesheet = pd.DataFrame({"path": files, "label": lables, "qscore": qscores, "read_id": readIDs, "aligned": aligned})
            if only_aligned:
                log.info(f"Filtering only aligned reads")
                filesheet = filesheet[filesheet["aligned"]]
            log.info(f"Found {len(filesheet)} files in total, aligned: {sum(filesheet['aligned'])}")

        min_n = filesheet.groupby("label").size().min()
        with log_time("Sub-sampling file-sheet", logger=log):
            if splitting.n_per_label > 0 and splitting.n_per_label < min_n:
                log.warning(f"There is more ({min_n}) samples per label than then max allowed ({splitting.n_per_label}). So sub-sampling all labels.")
                min_n = splitting.n_per_label
            filesheet = filesheet.groupby("label").sample(n=min_n, random_state=splitting.random_state)

        if split:
            with log_time("Splitting file-sheet into train and test", logger=log):
                train_filesheet, test_filesheet = train_test_split(
                    filesheet, test_size=splitting.test_size, random_state=splitting.random_state, stratify=filesheet["label"]
                )

        transform = []
        if preprocessing.rupture:
            transform.append(transforms.Rupture(preprocessing.rupture_part))
        if preprocessing.mad_normalize:
            transform.append(transforms.MADNormalize())
        if preprocessing.scale_to_fixed_length > 0:
            transform.append(transforms.ScaleToFixedLength(preprocessing.scale_to_fixed_length))
        if preprocessing.remove_outliers:
            transform.append(
                transforms.RemoveOutliers(std=preprocessing.remove_outliers_max_std, window_size=preprocessing.remove_outliers_window_size)
            )
        if preprocessing.left_padding > 0 or preprocessing.window_size > 0:
            transform.append(transforms.FixedWindow(preprocessing.window_size, leftpad=preprocessing.left_padding))
        if len(transform) > 0:
            transform = torchvision.transforms.Compose(transform)
        else:
            transform = None

        with log_time("Creating datasets", logger=log):
            if split:
                train_dataset = ReadDataset(
                    train_filesheet,
                    labels=label_order,
                    transform=transform,
                    comet_logger=comet_logger,
                    name=f"{train_dataset_path.stem}_train",
                )
                torch.save(train_dataset, train_dataset_path)
                test_dataset = ReadDataset(
                    test_filesheet,
                    labels=label_order,
                    transform=transform,
                    comet_logger=comet_logger,
                    name=f"{test_dataset_path.stem}_test",
                )
                torch.save(test_dataset, test_dataset_path)
                return train_dataset, test_dataset
            else:
                dataset = ReadDataset(
                    filesheet, labels=label_order, transform=transform, comet_logger=comet_logger, name=f"{dataset_path.stem}"
                )
                torch.save(dataset, dataset_path)
                return dataset
