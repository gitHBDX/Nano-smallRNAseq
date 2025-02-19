# Nano-smallRNAseq

This is the code for _Nanopore based RNA methylation profiling of a circulating lung cancer biomarker_.

## Installation

0. Make sure these tools are installed on your system, as they are called from inside the code:

- `rsync`
- `guppy_basecaller` (for SQK-RNA002)
- `dorado` (SQK-RNA004)
- `seqkit`
- `bwa`
- `samtools`

### Comet

1. Create an account and Project on https://www.comet.com/
2. Create a file `comet_api_key.txt` and copy your comet API key into it
3. Edit the `comet` section in the main config file `configs/main.yaml`

### Environment

4. Build the `nanopore` environment with `mamba` or `conda`

```
mamba create -n nanopore
mamba env update -n nanopore --file environment.yaml
mamba activate nanopore
```

## Training

### Configuration

Create config yaml files for your datasets used for training (so probably one for methylation and one for barcoding) in the folder `configs/input/inputset` following the example structure in `configs/input/inputset/example.yaml`. Each dataset is a collection of fast5 folders each given a label, a FASTA reference file and a name (for logging).

### Run training

To train a model run `python -m ont_hbdx.train`. No input data is set by default, so give a list of input data configs, which will be combined. You find and can add new datasets in `configs/input/inputset`.

Example with two data folder inputs (filenames without the suffix):

```bash
python -m ont_hbdx.train  +input/inputset=[barcoding_mir17,barcoding_miLung]
```

To run a grid-search over parameters that is part of the configs you can for example run: (Note the `--multirun` flag and the list of values for two params):

```bash
python -m ont_hbdx.train  +input/inputset=[230707_meth,230619_unmeth] input.min_qscore="0,3,7" splitting.n_per_label="1000,10000,100000" --multirun
```

## Testing

To run inference on an outside dataset, you can run `python -m ont_hbdx.test`. Set the datainput the same way as in training (no sampling, splitting will be done, even if configured). Further you give an experiment with the `e` config flag. Go on a comet.ml and find the experiment you want to load, then copy the experiment id (the long string of letters and numbers at the end of the URL) and pass it to the e flag. You can have multiple ones seperated by comma. right now the models are not saved online, so it only works with checkpoints that where run in the same folder.

```bash
python -m ont-hbdx.test  +input/inputset=[clinical_samples_1,****clinical_samples_2] e=?????
```

## Citation

> _Nanopore based RNA methylation profiling of a circulating lung cancer biomarker_
>
> Marta Sanchez-Delgado 1, Maurice Frank 1, Tomáš Šišmiš 2, Mustafa Kahraman 1, Alberto Daniel-Moreno 1, Emmika Mummery 1, Jessika Ceiler 1, Jasmin Skottke 1, Carla Bieg-Salazar 1, Franziska Hinkfoth 1, Christina Rudolf 1, Ronja Weiblen 1, Kaja Tikk 1, Tobias Sikosek 1, Bruno R Steinkraus 1, Rastislav Horos 1, Michal Urda 2, Timothy Rajakumar 1
>
> 1 Hummingbird Diagnostics GmbH, Heidelberg, Germany
>
> 2 Department of Pneumology and Phtiseology, University Hospital and Polyclinic F.D. Roosevelt Banská Bystrica, Banská Bystrica, Slovakia
