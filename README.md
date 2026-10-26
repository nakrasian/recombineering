# Recombinase discovery in *Xanthobacter*

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/nakrasian/recombineering/blob/main/Recombinase_Search/recombinase_results_explorer.py/wasm)

Discovery of single-strand annealing proteins (SSAPs) and their exonuclease
partners across 121 *Xanthobacter* genomes, aimed at oligonucleotide- and
dsDNA-based recombineering.

The search runs PSI-BLAST (Redβ / RecT / λ-exo seeds) and HMMER (Pfam SSAP and
YqaJ profiles) against the 121-genome protein set, deduplicates hits by pairwise
identity, and then looks for recombinase–exonuclease operons. Candidate systems
are cross-checked against a curated reference-phage panel scored by two
independent methods (BLASTN and PHASTEST).

**You do not need the search stack to read the results.** The workflows need
BLAST+, HMMER, MAFFT, skani and ~5 GB of genome FASTA; the results explorer needs
nothing but `pandas` and `altair`, because the answers are ~125 kB.

## Start here

| I want to… | Go to |
|---|---|
| Explore the results with no setup | [results explorer on molab](https://molab.marimo.io/github/nakrasian/recombineering/blob/main/Recombinase_Search/recombinase_results_explorer.py) |
| Interact with charts/plots | [WebAssembly preview](https://molab.marimo.io/github/nakrasian/recombineering/blob/main/Recombinase_Search/recombinase_results_explorer.py/wasm) (runs Python in your browser) |
| Re-run the search | [`Recombinase_Search/recombinase_workflow.py`](Recombinase_Search/recombinase_workflow.py) |
| Check a number | [`Recombinase_Search/share/data/manifest.json`](Recombinase_Search/share/data/manifest.json) — per-table source + SHA-256 |

## What's here

```
Recombinase_Search/
├── recombinase_workflow.py            # the search: PSI-BLAST + HMMER + dedup + operon calls
├── phage_recombinase_workflow.py      # reference-phage extension
├── recombinase_results_explorer.py    # read-only explorer (molab / WASM friendly)
├── share/
│   ├── build_bundle.py                # results/*.tsv + data/  ->  share/data/*.csv
│   ├── data/                          # 15 CSVs + manifest.json (125 kB) — the published results
│   └── README.md                      # rebuild rules, publishing recipes, what must not ship
├── results/                           # tabular results + figures from the analysis
├── __marimo__/session/                # committed notebook snapshots (static molab previews)
└── xanthobacter_121_accessions.txt    # the genome set, reproducible with the NCBI `datasets` CLI
```

Not published: genome FASTA, BLAST databases, PHASTEST output and other
multi-gigabyte inputs (see `.gitignore`), and the one-off `results/*.py` analysis
scripts, which hardcode local absolute paths.

## Headline numbers

| | |
|---|---|
| Genomes screened | 121 (14 species) |
| SSAP / exonuclease proteins found | 54 — RecT 26, YqaJ 15, ERF 10, Rad52 3 |
| Genomes with a co-located SSAP↔EXO operon | 11 |
| SSAP↔EXO pairs ≤ 5 kb | 12, all same-strand |
| Deduplicated bench candidates | 16 groups — RecT 6, YqaJ 5, ERF 3, Rad52 2 |

## Running it

Results explorer (any machine, no bioinformatics tooling):

```bash
cd Recombinase_Search
uv run --with marimo --with pandas --with altair \
  marimo edit recombinase_results_explorer.py
```

The full search, which needs `blast+`, `hmmer`, `mafft`, `skani`/`fastani` and the
NCBI `datasets` CLI on `PATH`, plus the genome set under `data/`:

```bash
cd Recombinase_Search
marimo edit recombinase_workflow.py
```

After the search finishes, refresh the published tables:

```bash
python share/build_bundle.py      # deterministic; commit share/data/
```

## Data & provenance

`share/data/manifest.json` records, per table, the source artifact, row and
column counts, byte size and SHA-256. The explorer surfaces those hashes in its
*Data provenance* tab, so a figure can be pinned to exact inputs. The bundle is
byte-reproducible: rebuilding from unchanged inputs produces identical files.

Genomes are public NCBI assemblies; the accession list is committed, so the input
set can be re-downloaded rather than redistributed.

## License

Not yet declared — add a `LICENSE` before this is reused outside the lab.
