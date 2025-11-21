This repository contains implementations and scripts for evaluating uncertainty quantification (UQ) methods on UCI regression datasets. The codebase includes:
- Conditional Diffusion Models (CDM): `cdm.py`,

This README explains how to set up the environment, run experiments locally, and submit jobs to the QUEST cluster. The Python scripts include baked-in parameters (see each file for exact defaults). The run scripts in `scripts/` demonstrate recommended SLURM settings for QUEST.

## Quick links

- Main training scripts: `cdm.py`

## Environment setup

These experiments were run with Conda environments. Example environment YAMLs are provided in `envs/`:

- `envs/env-sc-cqr-cdm.yaml` (CDM / conditional diffusion)

Recommended steps (local or on QUEST):

1. Install Miniconda / Anaconda on your machine or ensure Conda is available on QUEST.
2. Create and activate the environment that matches the experiment you want to run. Example:

```bash
# create the CDM environment
conda env create -f envs/env-sc-cqr-cdm.yaml
conda activate card_cdm
```
If you do not use Conda you can manually install the main dependencies (PyTorch or TensorFlow as required, numpy, scipy, pandas, scikit-learn). Use the environment YAMLs as a starting point.

Notes for QUEST:

- QUEST provides modules and GPUs; the `scripts/*.sh` job scripts assume you will activate a Conda environment inside the job (they use `eval "$(conda shell.bash hook)"` then `conda activate <env>`).
- Adjust `#SBATCH` headers (account/project, output paths, partition, time) to match your allocation.

## Running locally (interactive / development)

You can run each experiment script directly from the command line. Example using CDM:

```bash
# from repository root
conda activate card_cdm
python -u cdm.py --root /absolute/path/to/UCI_Datasets --dataset power-plant --epochs 100 --dropout 0.2 --run_cdm
```

For split/CQR variant:

```bash
conda activate card_cdm
python -u cdm_split_cqr.py --root /absolute/path/to/UCI_Datasets --dataset YearPredictionMSD --epochs 200 --dropout 0.15 --run_cdm
```

Notes:

- Use absolute paths in `--root` when running on cluster jobs to avoid ambiguity.
- Use `python -u` to get unbuffered stdout so logs appear live in job output files.
- Check the top of each Python file to see baked-in defaults. You can pass CLI args to override them when provided.

## Submitting jobs on SLURM

The `scripts/` directory contains example SLURM submission scripts which were used to run experiments on QUEST. They include recommended resource requests and environment activation steps. Customize them for your account and dataset paths.

Example: submit the CDM job

```bash
# make the script executable (once)
chmod +x scripts/run_cdm.sh

# submit it to SLURM
sbatch scripts/run_cdm.sh
```

What the job scripts do (high level):

- Load a clean module environment with `module purge` and set UTF-8
- Activate a Conda environment inside the batch script
- Set helpful thread-related env vars: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`
- Run the desired training script using `srun` and request GPUs with `--gres=gpu:a100:1`
- Redirect stdout/stderr to log files (paths are set in the `#SBATCH --output` and `--error` directives)

Customize these fields before running on QUEST:

- `#SBATCH --account=` — set to your QUEST allocation/project
- `#SBATCH --output` and `--error` — set to a directory you own (example uses `/home/xyz1234/logs/`)
- `--partition`, `--constraint`, and `--gres` — match the hardware you want (A100-SXM shown in examples)

Inspecting GPU in job logs

If you need to verify the job sees GPUs, the job scripts include a commented-out diagnostic snippet that prints CUDA availability and device names. Enable that snippet (remove comment markers) to make the job write GPU details to the .out log.

## Where parameters live

- Most script-level hyperparameters (epochs, dropout rates, dataset names) are provided via argparse CLI flags in the Python files but have defaults set in the source. Check the top of each Python file (`cdm.py`, `cdm_split_cqr.py`) to see the baked-in defaults.
- The `UCI_Datasets/*/data/` folders include ancillary files like `n_epochs.txt` and `dropout_rates.txt` that were used by experiments; those help reproduce previously reported runs.

## Logging, outputs and checkpoints

- SLURM scripts use `#SBATCH --output` and `--error` to capture stdout/stderr; change those to point to a logs folder in your home on QUEST (or a shared project folder).
- The Python scripts typically print progress to stdout and may save model checkpoints to disk. Search the Python files for `save`, `checkpoint`, or `torch.save` / `tf.train.Checkpoint` to find where models are persisted.

## Troubleshooting

- Conda activation fails in batch jobs: ensure your login shell supports conda and that `eval "$(conda shell.bash hook)"` is present in the script before `conda activate`.
- GPU not visible in job: check `--gres` and `--constraint` in your SBATCH header, and confirm your partition supports GPUs. Add the CUDA diagnostic snippet to the script to print device info.
- Missing packages: create the Conda environment from the provided YAMLs under `envs/` or install packages via pip inside the activated environment.

## Reproducing results

1. Pick a dataset under `UCI_Datasets/` and confirm the `--root`/`--dataset` path you will use.
2. Create/activate the matching Conda environment from `envs/`.
3. Run the script locally for a quick smoke test with reduced epochs or submit a job to QUEST using a `scripts/*.sh` wrapper.

## Next steps and optional improvements

- Add explicit output/checkpoint directory CLI args to each Python script to make experiment artifacts easier to collect.
- Add a small wrapper to collect results and summarize metrics after a job finishes.
- Add unit tests / smoke tests that run one epoch on a tiny synthetic dataset so users can validate the installation quickly.

---
