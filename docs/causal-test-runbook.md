# Running the causal test on the RTX 4090 (#37)

A checklist for running #37 on the university machine instead of the laptop. **What** is run
is fixed by `docs/preregistration-vocab.md` at the tag `prereg-37`; this page only covers
**how**. Commands are for Windows PowerShell; on Linux use `python3.11 -m venv .venv`,
`source .venv/bin/activate`, `sha256sum` instead of `Get-FileHash`, and `/` in paths.

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
