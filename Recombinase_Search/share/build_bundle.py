#!/usr/bin/env python3
"""Build the publishable results bundle for `recombinase_results_explorer.py`.

The full workflow needs BLAST+/HMMER/MAFFT/skani plus ~5 GB of genome inputs.
The *answers* are tiny: every table below is a few tens of kilobytes. This script
distills the workflow's heterogeneous artifacts (notebook TSVs, script dumps,
raw hmmsearch output, matrices, a human-readable report) into one tidy CSV bundle
plus a manifest, so a read-only explorer notebook can run anywhere -- molab,
WebAssembly, GitHub Pages, a static host -- with no binaries and no genome data.

Usage
-----
    python share/build_bundle.py            # rebuild share/data/
    python share/build_bundle.py --check    # rebuild and verify hashes change/stay

Outputs (share/data/):
    *.csv           tidy tables, UTF-8, comma-separated
    manifest.json   per-file provenance: source, rows, cols, bytes, sha256

Design rules
------------
* stdlib only -- this must run on any machine, before any venv exists.
* deterministic -- sorted rows, no timestamps in the tables themselves, so
  git diffs stay meaningful and molab previews stay reproducible.
* lossless where it matters, lossy where it doesn't -- drop absolute paths,
  drop empty/stale artifacts, keep every number.
* honest -- anything we cannot parse is counted in the manifest, not silently
  dropped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # .../Recombinase_Search
OUT = HERE / "data"

DATASET = "xanthobacter_121"
DS_DIR = ROOT / "data" / DATASET
RESULTS = ROOT / "results"


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _csv_bytes(header: list[str], rows: list[list[str]]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _cell(value: str) -> str:
    """Normalize one TSV cell: trim, and unwrap literal quotes around empties.

    The workflow writes some columns as `""` (a quoted empty string) rather than
    a truly empty field; without this, the CSV writer escapes those quotes again
    and the value round-trips as `""""""`.
    """
    v = value.strip()
    while len(v) >= 2 and v[0] == v[-1] == '"':
        v = v[1:-1].strip()
    return v


def read_tsv(path: Path) -> tuple[list[str], list[list[str]], int]:
    """Read a TSV that may start with free-text banner lines.

    Several of these files open with a human-readable header
    ("OPERON CO-LOCATION (...)", "====", a comment) before the real table.
    We locate the first line that has the modal column count and treat
    everything above it as preamble, and any later line with a different
    column count as a trailing summary section.

    Returns (header, rows, dropped_line_count).
    """
    lines = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")

    counts = [len(ln.split("\t")) for ln in lines]
    modal = max(set(counts), key=counts.count)
    if modal < 2:
        raise ValueError(f"{path} has no tabular block (modal column count {modal})")

    start = counts.index(modal)
    header = [_cell(c) for c in lines[start].split("\t")]
    rows, dropped = [], 0
    for ln in lines[start + 1 :]:
        cells = ln.split("\t")
        if len(cells) != modal:
            dropped += 1
            continue
        rows.append([_cell(c) for c in cells])
    return header, rows, dropped


def dedupe(header: list[str], rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Drop exact duplicate rows (merged hits repeat across seed searches)."""
    seen, out = set(), []
    for r in rows:
        key = tuple(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return header, out


# --------------------------------------------------------------------------- #
# the two artifacts that are not tables
# --------------------------------------------------------------------------- #
GENOME_HDR = re.compile(r"^###\s+(\S+):\s+(\d+)\s+SSAP,\s+(\d+)\s+exo")
FEATURE = re.compile(
    r"^\s*(SSAP|EXO)\s+(\S+)\s+(\S+?):\s*([\d,]+)-[\s]*([\d,]+)\s+\(([+-])\)"
    r"\s+(\S+)\s+(\S+)\s+E=(\S+)\s+(.*?)\s*$"
)
PAIR = re.compile(
    r"^\s*(\S+)\((\S+)\)\s+(\d+)-(\d+)\s+([+-])\s+<->\s+"
    r"(\S+)\((\S+)\)\s+(\d+)-(\d+)\s+([+-])\s+gap=(\d+)\s*bp\s+\[(\S+)\]"
)


def parse_operon_report(path: Path) -> tuple[list[dict], list[dict], dict]:
    """Turn the free-text co-location report into two tidy tables.

    The report has a per-genome section (parseable, one line per protein and one
    line per SSAP<->EXO pair) followed by summary sections ('bench shortlist',
    'Standalone exonucleases', 'Unplaced'). We parse the first and count the
    rest as unplaced_* so nothing is silently lost.
    """
    features: list[dict] = []
    pairs: list[dict] = []
    stats = {"sections": 0, "unparsed_lines": 0, "unplaced": [], "standalone": []}

    genome = None
    section = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.rstrip("\n")
        if not line.strip() or line.startswith("===="):
            continue
        if line.startswith("### "):
            m = GENOME_HDR.match(line)
            if m:
                genome = m.group(1)
                stats["sections"] += 1
            else:                                # a differently-shaped section
                genome = None
            continue
        if genome is None:
            continue
        if line.strip().startswith("--"):
            section = "pairs"
            continue
        if line.strip().startswith("(none)"):
            continue

        m = FEATURE.match(line)
        if m:
            role, family, contig, start, end, strand, locus, protein, ev, desc = m.groups()
            features.append({
                "genome": genome, "role": role, "family": family, "contig": contig,
                "start": start.replace(",", ""), "end": end.replace(",", ""),
                "strand": strand, "locus_tag": locus, "protein": protein,
                "evalue": ev, "description": desc,
            })
            continue

        m = PAIR.match(line)
        if m:
            (sf, sp, ss, se, sst, xf, xp, xs, xe, xst, gap, rel) = m.groups()
            pairs.append({
                "genome": genome,
                "ssap_family": sf, "ssap_protein": sp, "ssap_start": ss, "ssap_end": se,
                "ssap_strand": sst,
                "exo_family": xf, "exo_protein": xp, "exo_start": xs, "exo_end": xe,
                "exo_strand": xst, "gap_bp": gap, "relation": rel,
            })
            continue

        stats["unparsed_lines"] += 1

    # trailing free-text sections we deliberately do not tabulate
    text = path.read_text(encoding="utf-8", errors="replace")
    for name, key in (("Standalone exonucleases", "standalone"), ("Unplaced", "unplaced")):
        blob = text.split(name, 1)[1] if name in text else ""
        for ln in blob.splitlines()[1:]:
            parts = ln.split()
            if len(parts) >= 3 and parts[0].startswith("GC"):
                stats[key].append(parts[0])

    features.sort(key=lambda r: (r["genome"], r["role"], int(r["start"])))
    pairs.sort(key=lambda r: (r["genome"], int(r["gap_bp"])))
    return features, pairs, stats


def tidy_matrix(path: Path, index_name: str) -> tuple[list[str], list[list[str]]]:
    """Read a wide numeric matrix with a row-label column and blank cells."""
    header, rows, dropped = read_tsv(path)
    header = [index_name] + header[1:]
    clean = []
    for r in rows:
        if not r[0]:
            continue
        clean.append(r)
    if dropped:
        print(f"  ! {path.name}: dropped {dropped} line(s) that did not match the matrix")
    return header, clean


# --------------------------------------------------------------------------- #
# bundle definition -- the single source of truth for what gets published
# --------------------------------------------------------------------------- #
# (output name, title, description, loader)
def bundle_spec():
    """Each entry: filename stem -> (title, description, (header, rows), source)."""
    spec = []

    def add(name, title, desc, header, rows, source, extra=None):
        spec.append({
            "name": name, "title": title, "description": desc,
            "header": header, "rows": rows, "source": source, **(extra or {}),
        })

    # --- headline tables ---------------------------------------------------
    h, r, _ = read_tsv(RESULTS / "shortlist.tsv")
    add("shortlist", "Bench shortlist",
        "One representative per recombinase group, plus its redundant variants and "
        "distinct (70-90% identity) members. This is the list to synthesise.",
        h, r, "results/shortlist.tsv")

    h, r, _ = read_tsv(RESULTS / "species_recombinase_counts.tsv")
    add("species_counts", "Hits per species",
        "Genomes with hits and genomes with a co-located SSAP/exonuclease operon, "
        "per Xanthobacter species.", h, r, "results/species_recombinase_counts.tsv")

    h, r, _ = read_tsv(RESULTS / "per_genome_recombinase_counts.tsv")
    add("genome_counts", "Hits per genome",
        "Per-genome family counts and operon status for the 121-genome set.",
        h, r, "results/per_genome_recombinase_counts.tsv")

    # --- search evidence ---------------------------------------------------
    h, r, _ = read_tsv(RESULTS / "hmm_hits_consolidated_121.tsv")
    h = ["family", "genome", "protein", "evalue"] if len(h) == 4 else h
    add("hmm_hits", "HMMER hits (Pfam profiles)",
        "Per-protein Pfam family assignment and e-value for the 121-genome set.",
        h, r, "results/hmm_hits_consolidated_121.tsv")

    h, r, d = read_tsv(RESULTS / "psiblast_ssap_vs_121.tsv")
    add("psiblast_ssap", "PSI-BLAST hits, SSAP seeds",
        "Three-iteration PSI-BLAST of the Red_beta / RecT_Ecoli / Plu_beta seeds "
        "against the 121-genome protein set.", h, r, "results/psiblast_ssap_vs_121.tsv",
        {"note": f"{d} trailing line(s) dropped" if d else None})

    h, r, d = read_tsv(RESULTS / "psiblast_exo_vs_121.tsv")
    add("psiblast_exo", "PSI-BLAST hits, exonuclease seeds",
        "Three-iteration PSI-BLAST of the lambda_exo / RecE seeds against the "
        "121-genome protein set.", h, r, "results/psiblast_exo_vs_121.tsv",
        {"note": f"{d} trailing line(s) dropped" if d else None})

    # --- operon structure (parsed from the text report) --------------------
    feats, prs, stats = parse_operon_report(RESULTS / "co_location_121.tsv")
    fhdr = ["genome", "role", "family", "contig", "start", "end", "strand",
            "locus_tag", "protein", "evalue", "description"]
    add("operons", "Operon features",
        "Every SSAP/exonuclease protein found in the 121-genome set, with locus "
        "coordinates. Parsed from the workflow's co-location report.",
        fhdr, [[str(f[k]) for k in fhdr] for f in feats], "results/co_location_121.tsv",
        {"note": (f"{stats['sections']} genome sections parsed; "
                  f"{len(stats['standalone'])} standalone exonuclease(s) and "
                  f"{len(stats['unplaced'])} unplaced hit(s) listed in the report's "
                  f"text summaries are not tabulated")})

    phdr = ["genome", "ssap_family", "ssap_protein", "ssap_start", "ssap_end",
            "ssap_strand", "exo_family", "exo_protein", "exo_start", "exo_end",
            "exo_strand", "gap_bp", "relation"]
    add("operon_pairs", "SSAP / exonuclease pairs",
        "Recombinase-exonuclease pairs within 5 kb, with the intergenic gap and "
        "whether the two genes share a strand.",
        phdr, [[str(p[k]) for k in phdr] for p in prs], "results/co_location_121.tsv")

    # --- genome relatedness ------------------------------------------------
    h, r = tidy_matrix(DS_DIR / "ani" / "ani_matrix.tsv", "accession")
    add("ani_matrix", "ANI matrix (skani)",
        "Pairwise average nucleotide identity across the 121 genomes. Blank cells "
        "are pairs below skani's reporting threshold.",
        h, r, f"data/{DATASET}/ani/ani_matrix.tsv")

    h, r = tidy_matrix(RESULTS / "pairwise_identity_matrix.tsv", "protein")
    add("identity_matrix", "Recombinase % identity matrix",
        "MAFFT-derived pairwise percent identity between the representative "
        "recombinases, used for the deduplication clusters.",
        h, r, "results/pairwise_identity_matrix.tsv")

    # --- reference phages --------------------------------------------------
    h, r, _ = read_tsv(RESULTS / "ref_phage_recomb.tsv")
    add("ref_phage_recomb", "Reference phages vs Xanthobacter",
        "Per-phage recombination evidence against the Xanthobacter set: BLASTN "
        "regions, PHASTEST CDS hits, and the recombinase families encoded.",
        h, r, "results/ref_phage_recomb.tsv")

    h, r, _ = read_tsv(RESULTS / "ref_phage_collection.tsv")
    add("ref_phage_collection", "Reference phage collection",
        "The curated reference-phage panel scored by two independent methods "
        "(BLASTN against Xanthobacter, PHASTEST against the prophage set) plus the "
        "recombinase families each phage encodes. `both_methods` flags phages "
        "supported by both lines of evidence.",
        h, r, "results/ref_phage_collection.tsv")

    h, r, _ = read_tsv(RESULTS / "phastest_phage_hits_by_type.tsv")
    add("phage_hits_by_type", "PHASTEST hits by host type",
        "Aggregated PHASTEST hit counts grouped by phage host type.",
        h, r, "results/phastest_phage_hits_by_type.tsv")

    h, r, _ = read_tsv(RESULTS / "refseq_virus_vs_phastest.tsv")
    add("refseq_virus_vs_phastest", "RefSeq virus vs PHASTEST",
        "RefSeq viral genomes cross-referenced against the PHASTEST phage set.",
        h, r, "results/refseq_virus_vs_phastest.tsv")

    # --- metadata (path column stripped) -----------------------------------
    h, r, _ = read_tsv(DS_DIR / "meta" / "genomes.tsv")
    keep = ["accession", "source", "level", "contigs", "length", "species", "strain", "is_tiny"]
    idx = {c: i for i, c in enumerate(h)}
    rows = [[row[idx[c]] for c in keep] for row in r if all(c in idx for c in keep)]
    add("genomes", "Genome metadata",
        "Assembly metadata for the 121-genome set. The workflow's absolute "
        "local file paths are intentionally not published.",
        keep, rows, f"data/{DATASET}/meta/genomes.tsv")

    return spec


def build(check: bool = False) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = bundle_spec()

    manifest = {
        "dataset": DATASET,
        "generated_by": "share/build_bundle.py",
        "note": (
            "Read-only results bundle for recombinase_results_explorer.py. "
            "Regenerate with `python share/build_bundle.py` after the workflow "
            "changes; commit the CSVs so molab and WebAssembly previews stay in sync."
        ),
        "tables": [],
    }

    total = 0
    for entry in spec:
        header, rows = dedupe(entry["header"], entry["rows"])
        blob = _csv_bytes(header, rows)
        path = OUT / f"{entry['name']}.csv"
        old = path.read_bytes() if path.exists() else None
        path.write_bytes(blob)
        total += len(blob)

        changed = "new" if old is None else ("changed" if old != blob else "same")
        print(f"  {path.name:28s} {len(rows):5d} rows x {len(header):2d} cols "
              f"{len(blob) / 1024:7.1f} KiB  {changed}")

        manifest["tables"].append({
            "file": path.name,
            "title": entry["title"],
            "description": entry["description"],
            "source": entry["source"],
            "rows": len(rows),
            "cols": len(header),
            "header": header,
            "bytes": len(blob),
            "sha256": _sha256(blob),
            **(dict(note=entry["note"]) if entry.get("note") else {}),
        })

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\n  {len(spec)} tables, {total / 1024:.0f} KiB total -> {OUT.relative_to(ROOT)}/")
    if check:
        print("  manifest.json written; commit it alongside the CSVs.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report changes without failing")
    sys.exit(build(**vars(ap.parse_args())))
