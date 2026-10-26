# Publishable results bundle

`recombinase_results_explorer.py` (repo root) is a read-only companion notebook.
It renders the *answers* from `recombinase_workflow.py` without needing any of
the machinery that produced them.

This directory holds the distilled data it reads.

```
share/
├── build_bundle.py     # results/*.tsv + data/xanthobacter_121/* -> data/*.csv
├── data/               # committed: 15 CSVs (~126 kB) + manifest.json
│   ├── shortlist.csv
│   ├── ... (see manifest.json for the full list, sources, and SHA-256)
│   └── manifest.json
└── README.md
```

## Why a bundle instead of the raw artifacts

The workflow's outputs are heterogeneous: notebook TSVs, standalone script dumps,
raw `hmmsearch` output, wide numeric matrices, a few empty/stale files, and one
human-readable report (`results/co_location_121.tsv`). They also embed absolute
local paths (`/Users/...`) in `genomes.tsv`.

`build_bundle.py` normalizes all of that:

| Problem in the raw artifacts | What the bundle does |
|---|---|
| Banner/comment lines before the header (`OPERON CO-LOCATION (...)`) | Header detected by modal column count; preamble dropped |
| `""` quoted-empty cells re-escaped by CSV writers | Unwrapped on read |
| Absolute local paths in `genomes.tsv` | Column dropped — never published |
| Empty / stale artifacts (`blastp_redbeta_relaxed.tsv`, `psiblast_P03698.tsv`) | Excluded |
| Free-text operon report | Parsed into `operons.csv` + `operon_pairs.csv`; unparsed sections are counted in the manifest, not silently lost |
| Wide ANI / identity matrices with blank cells | Kept wide (smaller, melts in the notebook) |
| Exact duplicate rows from overlapping seed searches | Deduplicated |

Every file records its `source`, row/column counts, byte size, and SHA-256 in
`data/manifest.json`. The explorer surfaces those hashes in its **Data provenance**
tab, so a reader can pin the exact tables they looked at.

## Rebuild after the workflow changes

```bash
cd Recombinase_Search
python share/build_bundle.py          # stdlib only; no venv needed
git add share/data && git commit -m "results: refresh share bundle"
```

The builder is deterministic: unchanged inputs produce byte-identical CSVs, so
diffs show only what actually moved.

## Publish

### 1. Same repo as the workflow (simplest, recommended)

Commit `share/data/` and the explorer notebook, then push. Add the GitHub fallback
so the notebook works even with no checkout beside it — one line in the notebook's
`data_access` cell:

```python
GH_REPO = "your-org/your-repo"     # e.g. "sherbolab/recombineering"
GH_REF = "main"
SUBDIR = "Recombinase_Search"
```

Then molab gives you three share surfaces from one GitHub mirror
(<https://molab.marimo.io/github> generates the URLs):

| Surface | URL | Login for viewers | Runs |
|---|---|---|---|
| Static preview | `https://molab.marimo.io/github/<org>/<repo>/blob/main/Recombinase_Search/recombinase_results_explorer.py` | no | nothing (committed outputs) |
| Interactive / WebAssembly | append `/wasm` | no | Python in the viewer's browser (Pyodide) |
| Server-backed / “Run as app” | Share → *Run as app* | no for apps | real cloud compute (4 vCPU / 32 GB) |

Static previews only show outputs if the session snapshot is committed:

```bash
uv run --with marimo --with pandas --with altair \
  marimo export session recombinase_results_explorer.py
# writes __marimo__/session/recombinase_results_explorer.py.json — commit it
```

Badge for the README:

```markdown
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/your-org/your-repo/blob/main/Recombinase_Search/recombinase_results_explorer.py/wasm)
```

### 2. Hugging Face dataset (if the bundle outgrows the repo)

A public dataset repo is free, CDN-served, CORS-enabled, and versioned. Upload
`share/data/` and point the notebook at it by editing `_sources()` to yield
`https://huggingface.co/datasets/<user>/<name>/resolve/main/<file>`.

### 3. Zenodo (for the citable version)

Zip `share/data/` (plus the two notebooks) and deposit for a DOI. The bundle is
~130 kB, well under any repository limit, and `manifest.json` already contains the
hashes a reviewer would want.

## What WebAssembly can and cannot do here

The explorer is deliberately built to survive Pyodide:

* **Packages:** `pandas` + `altair` only — both supported in Pyodide. No polars
  (fine), no `pyarrow`, no `matplotlib`, no `subprocess`, no native extensions.
* **Memory:** WASM notebooks are capped at 2 GB; the bundle is 126 kB and the
  largest in-memory frame is the 121×121 ANI matrix.
* **Data access:** files are fetched over HTTPS, so the host must send CORS
  headers. `raw.githubusercontent.com` and `huggingface.co/.../resolve/...` both
  do; a private bucket does not.
* **No writes:** the notebook never writes to disk, so there is no need for
  molab's persistent storage or a remote-storage connection.

If you ever need a package WebAssembly cannot load, publish the *same* notebook
as a molab app (server-backed) instead — no code change, just a different link.

## What must never go in here

* the 2.5 GB `genomes/` FASTA, `db/proteins.faa`, BLAST indices, `phastest/`
  (13 GB) — inputs, not results;
* absolute paths, API keys, PHASTEST session cookies;
* anything you would not put on a public GitHub repo. molab notebooks are
  public-but-undiscoverable, which is gist-grade privacy, not security.
