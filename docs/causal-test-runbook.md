# Running the causal test on the RTX 4090 (#37)

A checklist for running #37 on an RTX 4090 instead of the laptop. **What** is run is fixed by
`docs/preregistration-vocab.md` at the tag `prereg-37`; this page only covers **how**.

- **Sections 1–6** were written for the university's desktop. Their commands are for Windows
  PowerShell. On Linux use `python3.11 -m venv .venv`, `source .venv/bin/activate`,
  `sha256sum` instead of `Get-FileHash`, and `/` in paths.
- **Section 7** covers the machine the test actually runs on: a rented RTX 4090 in a Linux
  container (deviation D1 of the preregistration).

## 1. Before going there

- [ ] **GPU and driver.** NVIDIA RTX 4090 (24 GB), driver recent enough for CUDA 12.1: version
      531 or newer on Windows, 530 or newer on Linux. `nvidia-smi` shows both.
- [ ] **Python 3.11 and Git** are installed, or you may install them as a user. Git is required:
      run IDs hash the git tree of `src/kazner`, so a copied folder without `.git` gives other
      run IDs and the NER configs will not find the pre-trained models.
- [ ] **Disk:** at least 20 GB free (virtual environment ~6 GB, data and models ~3 GB, one
      pre-training checkpoint ~2 GB at a time).
- [ ] **No sleep and no other GPU jobs** for about ten hours: power plan set to never sleep (or
      ask the administrator), and `nvidia-smi` lists no other processes.
- [ ] **Copied from the laptop** to a USB stick:
  - `data/kk_corpus/fineweb2_kaz_Cyrl.txt` and `fineweb2_kaz_Cyrl.manifest.json` (274 MiB);
  - the folder `models/mbert-kk-10k/` (~0.7 GB);
  - optionally `data/kaznerd/` (it can also be downloaded, step 2).

## 2. Setup

```powershell
git clone https://github.com/DiasZhaga/kazner.git
cd kazner
git checkout prereg-37
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[dev]"
python scripts/download_kaznerd.py     # checksum-verified; or copy data\kaznerd\
```

Copy the corpus to `data\kk_corpus\` and the extension to `models\mbert-kk-10k\`.

## 3. Checks before the first run

| Check | Command | Expected |
|---|---|---|
| GPU visible | `python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"` | `2.5.1+cu121 NVIDIA GeForce RTX 4090` |
| Clean checkout | `git status --short` | no output |
| Code | `git rev-parse HEAD:src/kazner` | `82c1f9dbdd253eb7e3c67ea7ae3fcd6790c3914b` |
| Corpus | `(Get-FileHash data\kk_corpus\fineweb2_kaz_Cyrl.txt).Hash` | `9AB7B0C56C74685E…74B0A` (section 6 of the preregistration) |
| Extended model | `python -c "from kazner.config import model_fingerprint as f; print(f('models/mbert-kk-10k'))"` | `661abc386fa31cbc…c7e9af` |
| Tests | `pytest -m "not network and not smoke" -q` | all pass |

## 4. Runs, in this order

```powershell
kazner pretrain --config configs/pretrain/mbert_kk_original.yaml   # writes models\4ea205f2243c
kazner pretrain --config configs/pretrain/mbert_kk_ext10k.yaml     # writes models\cd81fd453b74
foreach ($c in 'arm_a_full', 'arm_b_full', 'arm_a_lora', 'arm_b_lora', 'reference_full', 'reference_lora') {
    kazner train --config "configs/causal/$c.yaml"
}
```

- **In the first minute of each pre-training run,** check that it writes to the directory named
  in the comment. Another directory means different inputs or code: stop the run and find the
  difference before going on.
- **If something stops,** run the same command again. Complete runs are skipped. An interrupted
  pre-training run resumes from its last checkpoint (every 2,000 steps); an interrupted NER run
  starts again under the same run ID.
- **Do not open test metrics** (`metrics.json`, `predictions.csv`) until all 40 runs of arms A
  and B are complete (preregistration, section 5). The validation F1 in the console may be
  watched to catch failures.
- **Time.** Not measured on this GPU. The estimates are 30–60 min per pre-training run and
  4–8 min per NER run: batch 4 × 2 leaves much of the GPU idle. The 60 NER runs therefore take
  about 4–8 h.
- **Optional, after both pre-training runs:** to use the GPU better, run arm A's and arm B's
  NER configs in two terminals at the same time. Every run writes its own directory.

## 5. What to bring back

- `runs\`: all 60 run directories (config, environment, metrics, predictions; no weights).
- `models\4ea205f2243c\` and `models\cd81fd453b74\` (~0.7 GB each): the pre-trained arms, with
  their `pretrain_metrics.json` and `environment.json`.
- `pip freeze > pip-freeze-4090.txt` and the output of `nvidia-smi`, for the paper's methods
  section.

## 6. Analysis (on the laptop)

Run IDs per config:

```powershell
python -c "import json, pathlib, collections; d = collections.defaultdict(list); [d[json.loads(p.read_text(encoding='utf-8'))['name']].append(p.parent.name) for p in sorted(pathlib.Path('runs').glob('*/config.json'))]; [print(k, ','.join(v)) for k, v in sorted(d.items()) if k.startswith('causal_')]"
```

Then `kazner fragmentation` on the 60 run directories (`--data data/kaznerd --revision
3f076fdb1ab68d5b2880cb87a0886f315b8146f8 --out results/37`) and the `kazner compare` calls of
the preregistration, section 4: one primary call with exactly the pairs `full` and `lora` of
A against B, and a separate call for R against A.

## 7. On a rented RTX 4090 (Linux container)

The university machine is not available, so the test runs on a rented GPU (preregistration,
deviation D1). Everything in sections 3–6 still applies; this section adds the rental steps.

### 7.1 Renting

- **Provider.** RunPod or Vast.ai. Choose an **on-demand** instance: an interruptible ("spot")
  instance can be stopped mid-run.
- **Instance.** 1 × RTX 4090, 24 GB. Any Ubuntu 22.04 image whose driver supports CUDA 12.1 or
  later (driver 530 or newer). PyTorch is installed into a fresh virtual environment, so the
  image's own PyTorch does not matter.
- **Disk.** At least 40 GB of container disk.
- **Access.** SSH with your public key. A web terminal also works, but uploads are easier over
  SSH.
- **Record.** Write down the provider, the instance type and ID, and the price per hour. They go
  into deviation D1 after the runs.

Account, payment and the instance itself are created by the user.

### 7.2 Data to upload

One archive with the corpus, its manifest and the extended model, made on the laptop in the
main checkout (Windows has `tar`):

```powershell
tar -czf ..\upload37\causal37-data.tar.gz data/kk_corpus/fineweb2_kaz_Cyrl.txt `
    data/kk_corpus/fineweb2_kaz_Cyrl.manifest.json models/mbert-kk-10k
(Get-FileHash ..\upload37\causal37-data.tar.gz).Hash
scp -P <port> -i <key> ..\upload37\causal37-data.tar.gz root@<host>:/workspace/
```

KazNERD is downloaded on the instance by its checksum-verified script.

### 7.3 Setup on the instance

```bash
apt-get update && apt-get install -y git tmux python3.11 python3.11-venv  # if missing
cd /workspace
sha256sum causal37-data.tar.gz            # the hash printed on the laptop
git clone https://github.com/DiasZhaga/kazner.git && cd kazner
git checkout prereg-37
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[dev]"
python scripts/download_kaznerd.py
tar -xzf /workspace/causal37-data.tar.gz
```

Then the checks of section 3, with Linux commands. The corpus check is
`sha256sum data/kk_corpus/fineweb2_kaz_Cyrl.txt`. The GPU line must read
`2.5.1+cu121 NVIDIA GeForce RTX 4090`.

### 7.4 Runs

Start them inside `tmux`, so that closing the SSH session or the browser does not stop them.
The console output goes to a log file only: `kazner train` prints each run's **test** F1 when
the run ends, and the test results must not be looked at before the 40 runs of arms A and B are
complete (preregistration, section 5).

```bash
tmux new -s k37
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
mkdir -p runs/logs
( kazner pretrain --config configs/pretrain/mbert_kk_original.yaml &&
  kazner pretrain --config configs/pretrain/mbert_kk_ext10k.yaml &&
  for c in arm_a_full arm_b_full arm_a_lora arm_b_lora reference_full reference_lora; do
      kazner train --config configs/causal/$c.yaml
  done ) > runs/logs/causal37.log 2>&1
```

- **Detach and attach.** `Ctrl-b d` detaches; `tmux attach -t k37` comes back.
- **Progress without test scores:**
  - `ls -d models/4ea205f2243c models/cd81fd453b74` shows the pre-trained arms;
  - `ls runs/*/metrics.json | wc -l` counts the finished NER runs (60 at the end);
  - `grep -E "^\[run\]|eval_f1" runs/logs/causal37.log | tail` shows the current run and its
    validation F1.
- **First minutes.** Check that each pre-training run writes to the directory named in section
  4. If it does not, stop and compare the inputs.

### 7.5 Bringing results back and ending the rental

```bash
nvidia-smi > runs/logs/nvidia-smi-4090.txt
pip freeze > runs/logs/pip-freeze-4090.txt
tar -czf /workspace/causal37-results.tar.gz --exclude='*/checkpoints' \
    runs models/4ea205f2243c models/cd81fd453b74
```

```powershell
scp -P <port> -i <key> root@<host>:/workspace/causal37-results.tar.gz ..\upload37\
```

Unpack the archive in the main checkout and check that `runs/` holds 60 complete `causal_*`
runs. Only then **terminate** the instance. A stopped instance may still bill for its disk, and
terminating deletes everything on it.

### 7.6 Time and cost

The estimate of section 4 is 30–60 min per pre-training run and 4–8 h for the 60 NER runs,
about 5–10 h in all. At USD 0.35–0.90 per hour for an on-demand RTX 4090 (RunPod and Vast.ai,
autumn 2026), the test costs about USD 2–9. Running arm A's and arm B's NER configs in two
`tmux` windows after the pre-training (section 4) shortens the rental.
