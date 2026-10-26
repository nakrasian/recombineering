import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _intro(mo):
    mo.md(r"""
    # Phage Recombinase Discovery Workflow
    ## Overview
    This notebook mirrors `recombinase_workflow.py` but starts from the **prophage
    regions** detected in *Xanthobacter* genomes by **geNomad** and **PHASTEST**.
    Each prophage region is searched with **BLASTn** against the RefSeq-viral
    nucleotide database to identify the closest free-living phage; those hits are
    merged with the **PHASTEST database's own suggested phages** (`most_common_phage`),
    the union is **deduplicated by whole-genome % identity / coverage**, and the
    deduplicated representatives then go through the standard recombinase /
    exonuclease analysis (HMMER + PSI-BLAST, recombinase dedup, operon
    co-location, and a quality-weighted ranking).

    ### Outline:
    1. Import prophage regions from geNomad and PHASTEST (dedup near-identical regions)
    2. BLASTn each region → closest RefSeq-viral phage; compare against PHASTEST's suggestion
    3. Assemble the phage genome set (BLAST hits ∪ PHASTEST suggestions)
    4. **Deduplicate phages first** by whole-genome % identity / coverage; quality-ranked representatives
    5. Compile a protein database for the deduplicated phages (pyrodigal)
    6. Search for recombinases and exonucleases (HMMER + PSI-BLAST)
    7. Deduplicate recombinases by % similarity (MAFFT / BLOSUM62)
    8. Identify putative SSAP↔exonuclease/Gam operons
    9. Rank phages by genome quality + prophage-region support (+ systems present)

    ### Key Outputs:
    1. A ranked list of unique recombinases from the closest phages
    2. A ranked list of recombinase–exonuclease systems for dsDNA recombineering
    """)
    return


@app.cell
def _core_imports():

    import polars as pl
    import numpy as np
    import re
    import io
    import json
    import gzip
    import shutil
    import collections
    import subprocess
    import urllib.request
    import time
    from pathlib import Path
    import pandas as pd
    import altair as alt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from Bio import SeqIO, Entrez
    from Bio.Align import PairwiseAligner, substitution_matrices
    from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
    from scipy.spatial.distance import squareform
    from dna_features_viewer import GraphicFeature, GraphicRecord


    return (
        Entrez,
        GraphicFeature,
        GraphicRecord,
        PairwiseAligner,
        Path,
        SeqIO,
        alt,
        collections,
        dendrogram,
        gzip,
        io,
        json,
        linkage,
        matplotlib,
        np,
        pd,
        pl,
        plt,
        re,
        shutil,
        squareform,
        subprocess,
        substitution_matrices,
        time,
        urllib,
    )


@app.cell
def _pdf_util(matplotlib, mo, pd, plt):

    def pdf(df):
        """polars -> pandas for altair/plotly, avoiding the pyarrow path."""
        return pd.DataFrame(df.to_dicts())

    def result(obj, title="Result"):
        """Callout wrapper for a whole display block (charts / viewers)."""
        return mo.callout(obj, kind="info", title=title)

    def results_box(details, label="Results"):
        """Collapsed container for non-table output: logs, summaries, stats, context."""
        return mo.accordion({label: details})

    def matrix_heatmap_plotly(M, labels, vmin, vmax, cbar_title, title,
                              side_groups=None, blocks=None, height=620, width=None,
                              colorscale="Viridis"):
        """Interactive Plotly heatmap on a *numeric* (linear) axis."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        _n = len(labels)
        _pos = list(range(_n))
        _wid = width or int(height * 0.92) + (60 if _n > 45 else 0)
        _fs = max(6, min(11, int(260 / max(_n, 1)) + 5))
        _xr = [-0.5, _n - 0.5]
        _yr = [_n - 0.5, -0.5]
        _tick = dict(tickmode="array", tickvals=_pos, ticktext=[str(l) for l in labels],
                     tickfont=dict(size=_fs))
        _cd = [[[str(labels[i]), str(labels[j])] for j in _pos] for i in _pos]

        if side_groups is not None:
            _fig = make_subplots(rows=2, cols=2, column_widths=[0.03, 0.97],
                                 row_heights=[0.03, 0.97], horizontal_spacing=0.004,
                                 vertical_spacing=0.004)
            _xmain, _ymain, _row, _col = "x4", "y4", 2, 2
            _groups = list(dict.fromkeys(side_groups))
            _cmap = plt.get_cmap("tab20")
            _hex = {g: matplotlib.colors.to_hex(_cmap(i % 20)) for i, g in enumerate(_groups)}
            _gidx = {g: i for i, g in enumerate(_groups)}
            _zc = [_gidx[g] for g in side_groups]
            if len(_groups) == 1:
                _cs = [[0.0, _hex[_groups[0]]], [1.0, _hex[_groups[0]]]]
            else:
                _cs = [[i / (len(_groups) - 1), _hex[g]] for i, g in enumerate(_groups)]
            _fig.add_trace(go.Heatmap(z=[_zc], x=_pos, y=[0], colorscale=_cs,
                                      zmin=0, zmax=max(len(_groups) - 1, 1),
                                      showscale=False, hoverinfo="skip"), row=1, col=2)
            _fig.add_trace(go.Heatmap(z=[[v] for v in _zc], x=[0], y=_pos, colorscale=_cs,
                                      zmin=0, zmax=max(len(_groups) - 1, 1),
                                      showscale=False, hoverinfo="skip"), row=2, col=1)
        else:
            _fig = make_subplots(rows=1, cols=1)
            _xmain, _ymain, _row, _col = "x", "y", 1, 1

        _fig.add_trace(go.Heatmap(z=M, x=_pos, y=_pos, customdata=_cd,
                                  hovertemplate=("row %{customdata[0]} · col %{customdata[1]}"
                                                 "<br>%{z:.2f}<extra></extra>"),
                                  colorscale=colorscale, zmin=vmin, zmax=vmax,
                                  colorbar=dict(title=cbar_title, thickness=14, len=0.6)),
                       row=_row, col=_col)
        for (a, b, col, w) in (blocks or []):
            _fig.add_shape(type="rect", x0=a - 0.5, x1=b + 0.5, y0=a - 0.5, y1=b + 0.5,
                           xref=_xmain, yref=_ymain, line=dict(color=col, width=w),
                           fillcolor="rgba(0,0,0,0)", layer="above")

        _fig.update_layout(height=height, width=_wid, title=title, showlegend=False,
                           margin=dict(l=88, r=10, t=52, b=88), plot_bgcolor="white")
        _fig.update_layout(**{
            "xaxis" + _xmain[1:]: dict(**_tick, tickangle=90, showgrid=False, range=_xr),
            "yaxis" + _ymain[1:]: dict(**_tick, showgrid=False, range=_yr),
        })
        if side_groups is not None:
            _fig.update_layout(xaxis2=dict(showticklabels=False, showgrid=False, ticks="",
                                           range=_xr),
                               yaxis3=dict(showticklabels=False, showgrid=False, ticks="",
                                           range=_yr))
        return _fig


    return matrix_heatmap_plotly, pdf, result


@app.cell
def _parse_util():

    def _seed_id(qseqid):
        parts = qseqid.split("|")
        tok = parts[1] if len(parts) >= 2 else parts[0]
        return tok.split()[0]

    def parse_psiblast_tsv(path):
        """Parse BLAST/PSI-BLAST outfmt '6 std qcovhsp' into row dicts."""
        out = []
        with open(path) as _fh:
            for _line in _fh:
                if not _line.strip() or _line.startswith(("Search", "#")):
                    continue
                c = _line.rstrip("\n").split("\t")
                if len(c) < 12:
                    continue
                genome, _, protein = c[1].partition("|")
                try:
                    out.append({
                        "seed": _seed_id(c[0]), "genome": genome,
                        "protein": protein or c[1],
                        "pident": float(c[2]), "aln_len": int(c[3]),
                        "evalue": float(c[10]), "bitscore": float(c[11]),
                        "qcovhsp": int(c[12]) if len(c) > 12 else None,
                    })
                except ValueError:
                    continue
        return out

    def parse_blastn_tsv(path):
        """Parse blastn outfmt '6 qseqid sseqid pident length ... evalue bitscore qlen slen'."""
        out = []
        with open(path) as _fh:
            for _line in _fh:
                if not _line.strip() or _line.startswith(("#", "Search")):
                    continue
                c = _line.rstrip("\n").split("\t")
                if len(c) < 14:
                    continue
                try:
                    out.append({
                        "qseqid": c[0], "sseqid": c[1],
                        "pident": float(c[2]), "aln_len": int(c[3]),
                        "evalue": float(c[10]), "bitscore": float(c[11]),
                        "qlen": int(c[12]), "slen": int(c[13]),
                    })
                except ValueError:
                    continue
        return out


    return parse_blastn_tsv, parse_psiblast_tsv


@app.cell
def _config_ui(Path, mo):
    _root = Path(__file__).resolve().parent
    data_root = mo.ui.text(
        value=str(_root),
        label="Data root",
        full_width=True,
    )
    force_rebuild = mo.ui.switch(
        value=False,
        label="Force rebuild (overwrite cached artifacts)",
    )
    mo.vstack([data_root, force_rebuild])
    return data_root, force_rebuild


@app.cell
def _paths(Path, data_root, force_rebuild, mo, shutil):

    ROOT = Path(data_root.value).expanduser().resolve()
    OUT = ROOT / "data" / "phages"

    GENOMAD_DIR = ROOT / "genomad_out" / "xanthobacter_229_genomes_find_proviruses"
    GENOMAD_PROVIRUS_TSV = GENOMAD_DIR / "xanthobacter_229_genomes_provirus.tsv"
    GENOMAD_PROVIRUS_FNA = GENOMAD_DIR / "xanthobacter_229_genomes_provirus.fna"

    PHASTEST_DOCKER = ROOT / "phastest" / "phastest" / "phastest-app-docker" / "JOBS"
    PHASTEST_WEBAPI = ROOT / "phastest" / "webapi"

    REFSEQ_VIRAL_FNA = ROOT / "db" / "refseq_viral" / "viral.1.1.genomic.fna"
    REF_PHAGE_COLL = ROOT / "db" / "ref_phage_collection"
    REF_PHAGES = ROOT / "db" / "ref_phages"

    HMM_DIR = ROOT / "db" / "hmms"
    QUERY_DIR = ROOT / "db" / "queries"

    REGIONS_DIR = OUT / "regions"
    PHAGES_DIR = OUT / "phages"
    META_DIR = OUT / "meta"
    DB_DIR = OUT / "db"
    RESULTS_DIR = OUT / "results"
    MSA_DIR = OUT / "msa"

    REGIONS_FNA = REGIONS_DIR / "regions.fna"
    REGIONS_TSV = REGIONS_DIR / "regions.tsv"
    REGIONS_DEDUP_FNA = REGIONS_DIR / "regions_dedup.fna"
    REGIONS_DEDUP_TSV = REGIONS_DIR / "regions_dedup.tsv"

    VIRAL_BLAST_DB = DB_DIR / "refseq_viral_nt"
    REGION_VIRAL_BLAST_TSV = RESULTS_DIR / "blastn_region_vs_viral.tsv"

    PHAGES_FNA = PHAGES_DIR / "phages.fna"
    PHAGES_TSV = META_DIR / "phages.tsv"
    PHAGE_BLAST_DB = DB_DIR / "phages_nt"
    PHAGE_IDENTITY_TSV = RESULTS_DIR / "phage_identity.tsv"

    PROTEIN_FAA = DB_DIR / "proteins.faa"
    PROTEIN_BLAST_DB = DB_DIR / "proteins"
    GENES_TSV = DB_DIR / "genes.tsv"

    for _d in (REGIONS_DIR, PHAGES_DIR, META_DIR, DB_DIR, RESULTS_DIR, MSA_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    if force_rebuild.value:
        for _d in (REGIONS_DIR, META_DIR, DB_DIR, RESULTS_DIR, MSA_DIR):
            shutil.rmtree(_d, ignore_errors=True)
        for _d in (REGIONS_DIR, PHAGES_DIR, META_DIR, DB_DIR, RESULTS_DIR, MSA_DIR):
            _d.mkdir(parents=True, exist_ok=True)

    mo.md(
        f"**Data root:** `{ROOT}`  \n"
        f"**Output:** `{OUT}`  \n"
        + ("⚠️ `force_rebuild` **on** — cached artifacts cleared." if force_rebuild.value else "")
    )
    return (
        GENES_TSV,
        GENOMAD_PROVIRUS_FNA,
        GENOMAD_PROVIRUS_TSV,
        HMM_DIR,
        MSA_DIR,
        PHAGES_DIR,
        PHAGES_FNA,
        PHAGES_TSV,
        PHAGE_BLAST_DB,
        PHAGE_IDENTITY_TSV,
        PHASTEST_DOCKER,
        PHASTEST_WEBAPI,
        PROTEIN_BLAST_DB,
        PROTEIN_FAA,
        QUERY_DIR,
        REFSEQ_VIRAL_FNA,
        REF_PHAGES,
        REF_PHAGE_COLL,
        REGIONS_DEDUP_FNA,
        REGIONS_DEDUP_TSV,
        REGIONS_DIR,
        REGIONS_FNA,
        REGIONS_TSV,
        REGION_VIRAL_BLAST_TSV,
        RESULTS_DIR,
        VIRAL_BLAST_DB,
    )


@app.cell(hide_code=True)
def _step1_md(mo):
    mo.md(r"""
    ## Step 1 — Import prophage regions (geNomad + PHASTEST)

    Prophage regions come from two detectors, both already run on the *Xanthobacter*
    genome set:

    - **geNomad** (`genomad_out/…_find_proviruses/`): `provirus.tsv` (metadata) +
      `provirus.fna` (region nucleotide sequences).
    - **PHASTEST** (docker `JOBS/*/json_input_regions` + `region_DNA.txt`, and webapi
      `predicted_phage_regions.json` + `phage_regions.fna`). Both carry the same JSON:
      `region, start, stop, completeness, most_common_phage, GC`.

    The union is written to one `regions.tsv` with a `source` provenance column, and
    near-identical regions (the same prophage recurring across near-identical strains)
    are collapsed by sequence identity before BLAST.
    """)
    return


@app.cell
def _load_genomad(GENOMAD_PROVIRUS_FNA, GENOMAD_PROVIRUS_TSV, SeqIO, mo, pl):

    def load_genomad_regions():
        _rows = []
        if GENOMAD_PROVIRUS_TSV.exists():
            _df = pl.read_csv(GENOMAD_PROVIRUS_TSV, separator="\t", infer_schema_length=0)
            for _r in _df.iter_rows(named=True):
                _seq_name = _r["seq_name"]
                _genome, _contig, _prov = _seq_name.split("|")
                _rows.append({
                    "region_id": _seq_name, "source": "geNomad",
                    "genome": _genome, "contig": _contig,
                    "start": int(_r["start"]), "end": int(_r["end"]),
                    "length": int(_r["length"]), "n_genes": int(_r["n_genes"]),
                    "completeness": "", "most_common_phage": "",
                    "v_vs_c_score": float(_r.get("v_vs_c_score") or 0.0),
                })
        _seqs = {}
        if GENOMAD_PROVIRUS_FNA.exists():
            for _rec in SeqIO.parse(str(GENOMAD_PROVIRUS_FNA), "fasta"):
                _seqs[_rec.id] = str(_rec.seq)
        schema = {"region_id": pl.Utf8, "source": pl.Utf8, "genome": pl.Utf8,
                  "contig": pl.Utf8, "start": pl.Int64, "end": pl.Int64,
                  "length": pl.Int64, "n_genes": pl.Int64,
                  "completeness": pl.Utf8, "most_common_phage": pl.Utf8,
                  "v_vs_c_score": pl.Float64}
        _df = pl.DataFrame(_rows, schema=schema) if _rows else pl.DataFrame(schema=schema)
        return _df, _seqs

    genomad_regions_df, genomad_seqs = load_genomad_regions()
    log_genomad = mo.md(
        f"**geNomad**: {genomad_regions_df.height} prophage regions · "
        f"{len(genomad_seqs)} with nucleotide sequence · "
        f"{genomad_regions_df['genome'].n_unique() if genomad_regions_df.height else 0} genomes.")
    return genomad_regions_df, genomad_seqs, log_genomad


@app.cell
def _load_phastest(PHASTEST_DOCKER, PHASTEST_WEBAPI, SeqIO, json, mo, pl, re):

    def _norm_phage(name):
        _m = re.search(r"(NC_\d+)", name or "")
        return _m.group(1) if _m else (name or "")

    def _load_phastest_dir(_genome, _json_path, _fasta_path, _source):
        _rows = []
        if not _json_path.exists():
            return _rows, {}
        _data = json.loads(_json_path.read_text())
        _seqs = {}
        if _fasta_path.exists():
            for _rec in SeqIO.parse(str(_fasta_path), "fasta"):
                _seqs[str(_rec.id).strip()] = str(_rec.seq)
        for _d in _data:
            _rno = str(_d.get("region", ""))
            _rows.append({
                "region_id": f"{_genome}|{_source}|region_{_rno}",
                "source": f"PHASTEST-{_source}",
                "genome": _genome, "contig": "",
                "start": int(_d.get("start", 0)), "end": int(_d.get("stop", 0)),
                "length": int(abs(_d.get("stop", 0) - _d.get("start", 0))) + 1,
                "n_genes": None,
                "completeness": _d.get("completeness", ""),
                "most_common_phage": _d.get("most_common_phage", ""),
                "v_vs_c_score": None,
            })
        return _rows, _seqs

    def load_phastest_regions():
        _rows = []
        _seqs = {}
        if PHASTEST_DOCKER.exists():
            for _job in sorted(PHASTEST_DOCKER.iterdir()):
                if not _job.is_dir():
                    continue
                _jr = _job / "json_input_regions"
                _rd = _job / "region_DNA.txt"
                if _jr.exists():
                    _r, _s = _load_phastest_dir(_job.name, _jr, _rd, "docker")
                    _rows += _r
                    _seqs.update({_job.name + "|docker|region_" + k: v for k, v in _s.items()})
        if PHASTEST_WEBAPI.exists():
            for _acc_dir in sorted(PHASTEST_WEBAPI.iterdir()):
                if not _acc_dir.is_dir():
                    continue
                _jr = _acc_dir / "predicted_phage_regions.json"
                _pf = _acc_dir / "phage_regions.fna"
                if _jr.exists():
                    _r, _s = _load_phastest_dir(_acc_dir.name, _jr, _pf, "webapi")
                    _rows += _r
                    _seqs.update({_acc_dir.name + "|webapi|region_" + k: v for k, v in _s.items()})
        schema = {"region_id": pl.Utf8, "source": pl.Utf8, "genome": pl.Utf8,
                  "contig": pl.Utf8, "start": pl.Int64, "end": pl.Int64,
                  "length": pl.Int64, "n_genes": pl.Int64,
                  "completeness": pl.Utf8, "most_common_phage": pl.Utf8,
                  "v_vs_c_score": pl.Float64}
        _df = pl.DataFrame(_rows, schema=schema) if _rows else pl.DataFrame(schema=schema)
        return _df, _seqs

    phastest_regions_df, phastest_seqs = load_phastest_regions()
    _n_phage_hit = phastest_regions_df.filter(pl.col("most_common_phage") != "").height if phastest_regions_df.height else 0
    log_phastest = mo.md(
        f"**PHASTEST**: {phastest_regions_df.height} regions across "
        f"{phastest_regions_df['genome'].n_unique() if phastest_regions_df.height else 0} genomes · "
        f"{_n_phage_hit} with a `most_common_phage` suggestion.")
    return log_phastest, phastest_regions_df, phastest_seqs


@app.cell
def _regions_merge(
    REGIONS_FNA,
    REGIONS_TSV,
    genomad_regions_df,
    genomad_seqs,
    mo,
    phastest_regions_df,
    phastest_seqs,
    pl,
):

    regions_df = pl.concat([genomad_regions_df, phastest_regions_df], how="vertical")
    region_seqs = {}
    region_seqs.update(genomad_seqs)
    region_seqs.update(phastest_seqs)

    with open(REGIONS_FNA, "w") as _fh:
        for _r in regions_df.iter_rows(named=True):
            _s = region_seqs.get(_r["region_id"])
            if _s:
                _fh.write(f">{_r['region_id']}\n{_s}\n")
    regions_df.write_csv(REGIONS_TSV, separator="\t")

    _n_seq = sum(1 for _row in regions_df.iter_rows(named=True)
                 if _row.get("region_id") in region_seqs)
    log_regions = mo.md(
        f"**Union**: {regions_df.height} regions "
        f"(geNomad {genomad_regions_df.height} + PHASTEST {phastest_regions_df.height}) · "
        f"{_n_seq} with nucleotide sequence · written to `{REGIONS_TSV.name}`.")
    return log_regions, region_seqs, regions_df


@app.cell(hide_code=True)
def _step2_md(mo):
    mo.md(r"""
    ## Step 2 — BLASTn each region → closest RefSeq-viral phage

    Each distinct prophage region is queried with **BLASTn** (dc-megablast, for
    divergent matches) against the RefSeq-viral nucleotide database
    (`db/refseq_viral/viral.1.1.genomic.fna`, 19,625 viral genomes). The top hit per
    region is reported with % identity, alignment coverage, bitscore and E-value.

    Because prophage regions are chimeric (host + phage genes), coverage is often low;
    that is *expected*, and the BLAST top-hit is compared side-by-side with PHASTEST's
    own `most_common_phage` suggestion in the comparison table below — the two methods
    frequently disagree, which is itself informative.
    """)
    return


@app.cell
def _region_dedup_controls(mo):
    reg_id_slider = mo.ui.slider(80, 100, step=1, value=99,
                                 label="Region duplicate % identity ≥", show_value=True)
    reg_cov_slider = mo.ui.slider(0.5, 1.0, step=0.05, value=0.9,
                                  label="Region duplicate coverage ≥", show_value=True)
    mo.hstack([reg_id_slider, reg_cov_slider], justify="start", gap=2)
    return reg_cov_slider, reg_id_slider


@app.cell
def _region_dedup(
    Path,
    REGIONS_DEDUP_FNA,
    REGIONS_DEDUP_TSV,
    REGIONS_DIR,
    REGIONS_FNA,
    collections,
    force_rebuild,
    mo,
    parse_blastn_tsv,
    pl,
    reg_cov_slider,
    reg_id_slider,
    region_seqs,
    regions_df,
    subprocess,
):

    def _run_region_blast():
        _db = REGIONS_DIR / "regions_nt"
        _tsv = REGIONS_DIR / "regions_self.tsv"
        if not (Path(str(_db) + ".ndb").exists() or Path(str(_db) + ".nin").exists()) or force_rebuild.value:
            subprocess.run(["makeblastdb", "-in", str(REGIONS_FNA), "-dbtype", "nucl",
                            "-out", str(_db)], capture_output=True, text=True)
        _r = subprocess.run(
            ["blastn", "-task", "megablast", "-db", str(_db), "-query", str(REGIONS_FNA),
             "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend "
                        "sstart send evalue bitscore qlen slen",
             "-evalue", "1e-10", "-max_target_seqs", "100000", "-num_threads", "4",
             "-out", str(_tsv)], capture_output=True, text=True)
        return _tsv, f"region self-blastn rc={_r.returncode}"

    def _cluster_from_blast(_tsv, _thr_id, _thr_cov):
        _adj = collections.defaultdict(set)
        _ids = list(region_seqs.keys())
        _idx = {s: i for i, s in enumerate(_ids)}
        for _row in parse_blastn_tsv(_tsv):
            if _row["qseqid"] == _row["sseqid"]:
                continue
            if _row["qseqid"] not in _idx or _row["sseqid"] not in _idx:
                continue
            _cov_q = _row["aln_len"] / _row["qlen"] if _row["qlen"] else 0.0
            _cov_s = _row["aln_len"] / _row["slen"] if _row["slen"] else 0.0
            _cov = min(_cov_q, _cov_s)
            if _row["pident"] >= _thr_id and _cov >= _thr_cov:
                _i, _j = _idx[_row["qseqid"]], _idx[_row["sseqid"]]
                _adj[_i].add(_j); _adj[_j].add(_i)
        _seen, _comps = set(), []
        for _i in range(len(_ids)):
            if _i in _seen:
                continue
            _stack, _comp = [_i], []
            while _stack:
                _x = _stack.pop()
                if _x in _seen:
                    continue
                _seen.add(_x); _comp.append(_x); _stack.extend(_adj[_x] - _seen)
            _comps.append(sorted(_comp))
        return _comps, _ids

    _tsv, _log = _run_region_blast()
    _comps, _ids = _cluster_from_blast(_tsv, reg_id_slider.value, reg_cov_slider.value)
    _lengths = {s: len(region_seqs[s]) for s in _ids}
    _reps = [min(_comp, key=lambda i: (-_lengths[_ids[i]], _ids[i])) for _comp in _comps]
    rep_ids = [_ids[i] for i in _reps]

    _dedup_rows = []
    for _comp, _rep_i in zip(_comps, _reps):
        _dedup_rows.append({
            "region_id": _ids[_rep_i], "n_members": len(_comp),
            "members": ",".join(_ids[i] for i in _comp),
        })
    dedup_regions_df = (pl.DataFrame(_dedup_rows).sort("n_members", descending=True)
                        if _dedup_rows else pl.DataFrame(schema={"region_id": pl.Utf8,
                                                                 "n_members": pl.Int64,
                                                                 "members": pl.Utf8}))

    with open(REGIONS_DEDUP_FNA, "w") as _fh:
        for _rid in rep_ids:
            _fh.write(f">{_rid}\n{region_seqs[_rid]}\n")
    dedup_regions_df.write_csv(REGIONS_DEDUP_TSV, separator="\t")

    region_meta = {_r["region_id"]: _r for _r in regions_df.iter_rows(named=True)}
    dedup_meta = pl.DataFrame([
        dict(region_meta.get(_rid, {}), region_id=_rid,
             source=region_meta.get(_rid, {}).get("source", "?"))
        for _rid in rep_ids])

    log_dedup = mo.md(
        f"**{len(_ids)} regions → {len(rep_ids)} distinct** at ≥{reg_id_slider.value}% id / "
        f"≥{reg_cov_slider.value} cov · largest cluster "
        f"{max(dedup_regions_df['n_members']) if dedup_regions_df.height else 0} members · {_log}")
    return dedup_meta, log_dedup, rep_ids


@app.cell
def _blastn_config(mo):
    blastn_run_button = mo.ui.run_button(label="⟳ Re-run BLASTn (regions vs RefSeq viral)")
    mo.hstack([blastn_run_button], justify="start")
    return (blastn_run_button,)


@app.cell
def _blastn_load(
    Path,
    REFSEQ_VIRAL_FNA,
    REGIONS_DEDUP_FNA,
    REGION_VIRAL_BLAST_TSV,
    VIRAL_BLAST_DB,
    blastn_run_button,
    force_rebuild,
    mo,
    parse_blastn_tsv,
    pl,
    rep_ids,
    subprocess,
):

    def _build_viral_db():
        _idx = Path(str(VIRAL_BLAST_DB) + ".nin")
        if _idx.exists() and not force_rebuild.value:
            return "RefSeq-viral BLAST DB cached"
        if not REFSEQ_VIRAL_FNA.exists():
            return "⚠️ RefSeq viral FASTA missing"
        _r = subprocess.run(["makeblastdb", "-in", str(REFSEQ_VIRAL_FNA),
                             "-dbtype", "nucl", "-out", str(VIRAL_BLAST_DB),
                             "-title", "RefSeq viral nt"],
                            capture_output=True, text=True)
        return f"makeblastdb (viral) rc={_r.returncode}"

    def _run_blastn_regions(_do_rerun):
        _log = [_build_viral_db()]
        if _do_rerun or force_rebuild.value or not REGION_VIRAL_BLAST_TSV.exists():
            _r = subprocess.run(
                ["blastn", "-task", "dc-megablast", "-db", str(VIRAL_BLAST_DB),
                 "-query", str(REGIONS_DEDUP_FNA),
                 "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend "
                            "sstart send evalue bitscore qlen slen",
                 "-evalue", "1e-5", "-max_target_seqs", "1", "-num_threads", "4",
                 "-out", str(REGION_VIRAL_BLAST_TSV)], capture_output=True, text=True)
            _log.append(f"blastn rc={_r.returncode}")
        _rows = parse_blastn_tsv(REGION_VIRAL_BLAST_TSV) if REGION_VIRAL_BLAST_TSV.exists() else []
        _df = pl.DataFrame(_rows) if _rows else pl.DataFrame(
            schema={"qseqid": pl.Utf8, "sseqid": pl.Utf8, "pident": pl.Float64,
                    "aln_len": pl.Int64, "evalue": pl.Float64, "bitscore": pl.Float64,
                    "qlen": pl.Int64, "slen": pl.Int64})
        return _df, _log

    region_blast_df, _log = _run_blastn_regions(blastn_run_button.value)
    mo.md(f"Parsed **{region_blast_df.height} region → phage** top hits "
          f"from {len(rep_ids)} representative region(s)"
          + (f"  \n{'; '.join(_log)}" if _log else ""))
    return (region_blast_df,)


@app.cell
def _blast_vs_phastest_compare(dedup_meta, mo, pl, re, region_blast_df):

    ph_sugg = {}
    for _r in dedup_meta.iter_rows(named=True):
        _m = re.search(r"(NC_\d+)", _r.get("most_common_phage") or "")
        if _m:
            ph_sugg[_r["region_id"]] = _m.group(1)
    blast_hit = {_r["qseqid"]: _r["sseqid"] for _r in region_blast_df.iter_rows(named=True)}

    _all_regions = sorted(set(list(blast_hit.keys()) + list(ph_sugg.keys())))
    _rows = []
    for _rid in _all_regions:
        _b = blast_hit.get(_rid, "")
        _p = ph_sugg.get(_rid, "")
        _rows.append({
            "region_id": _rid,
            "blastn_top_phage": _b,
            "phastest_phage": _p,
            "match": "Y" if (_b and _p and _b == _p) else ("N" if _b and _p else ""),
        })
    compare_df = (pl.DataFrame(_rows).sort("region_id") if _rows
                  else pl.DataFrame(schema={"region_id": pl.Utf8, "blastn_top_phage": pl.Utf8,
                                            "phastest_phage": pl.Utf8, "match": pl.Utf8}))
    _n_both = compare_df.filter((pl.col("blastn_top_phage") != "") & (pl.col("phastest_phage") != "")).height
    _n_match = compare_df.filter(pl.col("match") == "Y").height if compare_df.height else 0
    sum_compare = mo.md(
        f"**{compare_df.height} regions** compared · {_n_both} with both signals · "
        f"**{_n_match} agree** between BLASTn top-hit and PHASTEST suggestion "
        f"({100.0 * _n_match / _n_both if _n_both else 0:.0f}% of comparable).")
    mo.ui.table(compare_df, selection=None, pagination=True, page_size=15)
    return compare_df, sum_compare


@app.cell(hide_code=True)
def _step3_md(mo):
    mo.md(r"""
    ## Step 3 — Assemble the phage genome set

    The candidate phages are the **union** of (a) each region's BLASTn top-hit and
    (b) PHASTEST's `most_common_phage` suggestions — accession-normalised to
    `NC_xxxxxx`. Genomes are fetched from NCBI (`efetch`, cached) and reused from the
    already-downloaded `db/ref_phage_collection/` and `db/ref_phages/` where the
    accession matches. Metadata (name, length, contig count) is parsed from the FASTA.
    """)
    return


@app.cell
def _phage_set_build(
    PHAGES_DIR,
    PHAGES_FNA,
    PHAGES_TSV,
    REF_PHAGES,
    REF_PHAGE_COLL,
    SeqIO,
    compare_df,
    mo,
    pl,
    re,
    region_blast_df,
    shutil,
    time,
    urllib,
):

    def _norm_acc(_a):
        _m = re.search(r"(NC_\d+)(?:\.\d+)?", _a or "")
        return _m.group(1) if _m else ""

    blast_accs = {_norm_acc(_r["sseqid"]) for _r in region_blast_df.iter_rows(named=True)}
    phastest_accs = set()
    for _r in compare_df.iter_rows(named=True):
        _a = _norm_acc(_r["phastest_phage"])
        if _a:
            phastest_accs.add(_a)
    all_accs = sorted(blast_accs | phastest_accs)

    def _fetch(_acc, _dest):
        _url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                f"?db=nuccore&id={_acc}&rettype=fasta&retmode=text")
        for _attempt in range(4):
            try:
                with urllib.request.urlopen(urllib.request.Request(_url), timeout=60) as _r:
                    _text = _r.read().decode("utf-8", "replace")
                    if _text.startswith(">"):
                        _dest.write_text(_text)
                        return True
            except Exception:
                time.sleep(1.5 * (_attempt + 1))
        return False

    _fetch_log = []
    for _acc in all_accs:
        _dest = PHAGES_DIR / f"{_acc}.fna"
        if _dest.exists() and _dest.stat().st_size > 0:
            continue
        for _src_dir in (REF_PHAGE_COLL, REF_PHAGES):
            _src = _src_dir / f"{_acc}.fna"
            if _src.exists() and _src.stat().st_size > 0:
                shutil.copy(str(_src), str(_dest))
                break
        if not _dest.exists() or _dest.stat().st_size == 0:
            _ok = _fetch(_acc, _dest)
            if _ok:
                _fetch_log.append(f"fetched {_acc}")
            else:
                _fetch_log.append(f"⚠️ failed {_acc}")
            time.sleep(0.4)

    _rows = []
    for _acc in all_accs:
        _dest = PHAGES_DIR / f"{_acc}.fna"
        if not _dest.exists() or _dest.stat().st_size == 0:
            _rows.append({"accession": _acc, "name": "", "length": None, "n_records": 0,
                          "in_blastn": _acc in blast_accs, "in_phastest": _acc in phastest_accs,
                          "source": "missing"})
            continue
        _recs = list(SeqIO.parse(str(_dest), "fasta"))
        _length = sum(len(_rec.seq) for _rec in _recs)
        _first = _recs[0].description if _recs else ""
        _rows.append({
            "accession": _acc, "name": _first.split(",")[0] if _first else "",
            "length": _length, "n_records": len(_recs),
            "in_blastn": _acc in blast_accs, "in_phastest": _acc in phastest_accs,
            "source": "RefSeq" if ("ref|" in (_recs[0].id if _recs else "") or _acc.startswith("NC_")) else "GenBank",
        })
    phage_meta_df = pl.DataFrame(_rows).sort("length", descending=True, nulls_last=True)
    phage_meta_df.write_csv(PHAGES_TSV, separator="\t")

    with open(PHAGES_FNA, "w") as _fh:
        for _acc in all_accs:
            _p = PHAGES_DIR / f"{_acc}.fna"
            if _p.exists():
                _fh.write(_p.read_text())

    log_phage_set = mo.md(
        f"**{len(all_accs)} candidate phages** ({len(blast_accs)} from BLASTn ∪ "
        f"{len(phastest_accs)} from PHASTEST; {len(blast_accs & phastest_accs)} overlap) · "
        + ("; ".join(_fetch_log) if _fetch_log else "all genomes cached/reused"))
    mo.ui.table(phage_meta_df, selection=None, pagination=True, page_size=15)
    return log_phage_set, phage_meta_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3b — Phage host phylogeny (what they infect & relation to *X. autotrophicus*)

    Each closest phage's **known host genus** is read from its NCBI genome title
    (e.g. `Rhizobium phage RHph_Y65`); phages whose title names no host are excluded.
    The host genera are mapped onto the **NCBI Taxonomy** hierarchy (a known taxonomic
    tree) and rendered as a horizontal dendrogram with *Xanthobacter autotrophicus*
    placed among them — showing whether the closest free-living phages infect organisms
    related to the *Xanthobacter* prophage source.
    """)
    return


@app.cell
def _(Entrez, RESULTS_DIR, mo, phage_meta_df, re, time):
    def _host_genus(name):
        _m = re.search(r"([A-Z][a-z]+)\s+(?:phage|virus|sp\.)", name or "")
        return _m.group(1) if _m else ""

    host_of = {}
    for _r in phage_meta_df.iter_rows(named=True):
        host_of[_r["accession"]] = _host_genus(_r["name"])

    host_genuses = sorted({g for g in host_of.values() if g})

    _TAX_CACHE = RESULTS_DIR / "host_taxonomy.tsv"
    _cached = {}
    if _TAX_CACHE.exists():
        for _line in _TAX_CACHE.read_text().splitlines():
            _p = _line.split("\t")
            if len(_p) >= 2:
                _cached[_p[0]] = _p[1]

    Entrez.email = "marimo@example.org"
    _want = sorted(set(host_genuses) | {"Xanthobacter"})
    _missing = [g for g in _want if g not in _cached]
    for _g in _missing:
        try:
            _h = Entrez.esearch(db="taxonomy", term=f"{_g}[Scientific Name]", retmax=1)
            _ids = Entrez.read(_h)["IdList"]; _h.close()
            _lin = ""
            if _ids:
                _h = Entrez.efetch(db="taxonomy", id=_ids[0])
                _rec = Entrez.read(_h); _h.close()
                _lin = _rec[0]["Lineage"]
            _cached[_g] = _lin
        except Exception:
            _cached[_g] = ""
        time.sleep(0.4)
    if _missing:
        with open(_TAX_CACHE, "w") as _fh:
            for _k in sorted(_cached):
                _fh.write(f"{_k}\t{_cached[_k]}\n")

    host_lineage = {_g: [t.strip() for t in _cached.get(_g, "").split(";") if t.strip()]
                    for _g in _want}

    log_hosts = mo.md(
        f"**Host genera:** {len(host_genuses)} across "
        f"{sum(1 for v in host_of.values() if v)} phages "
        f"(ignored {sum(1 for v in host_of.values() if not v)} without a named host) · "
        f"NCBI Taxonomy lineages {'fetched' if _missing else 'cached'} for "
        f"{len(host_lineage)} taxa.")
    return host_lineage, host_of, log_hosts


@app.cell
def _(dendrogram, host_lineage, host_of, linkage, mo, np, plt, squareform):
    def _shared_prefix(a, b):
        _n = 0
        for _x, _y in zip(a, b):
            if _x == _y:
                _n += 1
            else:
                break
        return _n

    _host_counts = {}
    for _a, _g in host_of.items():
        if _g:
            _host_counts[_g] = _host_counts.get(_g, 0) + 1

    _tree_genera = sorted(set(list(_host_counts.keys()) + ["Xanthobacter"]))
    _lin = [host_lineage.get(g, []) for g in _tree_genera]
    _n = len(_tree_genera)
    _D = np.zeros((_n, _n))
    for _i in range(_n):
        for _j in range(_i + 1, _n):
            _sp = _shared_prefix(_lin[_i], _lin[_j])
            _D[_i, _j] = _D[_j, _i] = max((len(_lin[_i]) - _sp) + (len(_lin[_j]) - _sp), 1)

    _apha = [("Alphaproteobacteria" in _lin[_i]) for _i in range(_n)]

    if _n >= 2:
        _Z = linkage(squareform(_D), method="average")
        _fig, _ax = plt.subplots(figsize=(11, max(6.0, 0.34 * _n)))
        _dn = dendrogram(_Z, orientation="left", no_labels=True, ax=_ax,
                         link_color_func=lambda k: "#999999")
        _order = _dn["leaves"]
        _ax.set_yticks([5 + 10 * i for i in range(_n)])
        _ax.set_yticklabels([f"{_tree_genera[i]} ({_host_counts.get(_tree_genera[i], 0)})"
                             for i in _order])
        for _tick, _i in zip(_ax.get_yticklabels(), _order):
            if _tree_genera[_i] == "Xanthobacter":
                _tick.set_color("#d62728"); _tick.set_fontweight("bold")
            elif _apha[_i]:
                _tick.set_color("#1f77b4")
            else:
                _tick.set_color("#555555")
        _ax.yaxis.tick_right()
        _ax.tick_params(axis="y", length=0)
        for _sp in ("top", "right", "bottom", "left"):
            _ax.spines[_sp].set_visible(False)
        _ax.set_xlabel("taxonomic distance (ranks apart in NCBI Taxonomy)")
        _ax.set_title("Phage host phylogeny — Xanthobacter (red), "
                      "other Alphaproteobacteria (blue)", fontsize=10)
        _fig.tight_layout()
        _out = _fig
    else:
        _out = mo.md("*Not enough host lineages to build a tree.*")
    _out
    return


@app.cell(hide_code=True)
def _step4_md(mo):
    mo.md(r"""
    ## Step 4 — Deduplicate phages FIRST (whole-genome % identity / coverage)

    Before any recombinase analysis, the phage genomes are deduplicated: an all-vs-all
    **BLASTn** (megablast) yields pairwise **% identity** and **% coverage** (both
    symmetrised by taking the minimum of the two directions, exactly like the main
    notebook's ANI/AF). Near-identical phages are clustered (single-linkage) and a
    **quality-ranked representative** is chosen per cluster:

    `RefSeq > GenBank → single-record > multi → longer genome → more CDS`

    Only the representatives proceed to the recombinase search (Step 6).
    """)
    return


@app.cell
def _phage_identity(
    PHAGES_FNA,
    PHAGE_BLAST_DB,
    PHAGE_IDENTITY_TSV,
    Path,
    force_rebuild,
    mo,
    np,
    parse_blastn_tsv,
    phage_meta_df,
    re,
    subprocess,
):

    def _build_nt_db():
        _idx = Path(str(PHAGE_BLAST_DB) + ".nin")
        if _idx.exists() and not force_rebuild.value:
            return "phage nt BLAST DB cached"
        if not PHAGES_FNA.exists():
            return "⚠️ phages.fna missing"
        _r = subprocess.run(["makeblastdb", "-in", str(PHAGES_FNA), "-dbtype", "nucl",
                             "-out", str(PHAGE_BLAST_DB)], capture_output=True, text=True)
        return f"makeblastdb (phages) rc={_r.returncode}"

    def _norm_blast_acc(_x):
        _m = re.search(r"(NC_\d+)", _x)
        return _m.group(1) if _m else _x

    def _matrices():
        _accs = sorted(phage_meta_df["accession"].to_list())
        _idx = {a: i for i, a in enumerate(_accs)}
        _n = len(_accs)
        _qlen = {_r["accession"]: _r["length"] or 1 for _r in phage_meta_df.iter_rows(named=True)}
        _best = {}
        for _row in parse_blastn_tsv(PHAGE_IDENTITY_TSV):
            _qa = _norm_blast_acc(_row["qseqid"].split()[0])
            _sa = _norm_blast_acc(_row["sseqid"].split()[0])
            if _qa not in _idx or _sa not in _idx or _qa == _sa:
                continue
            _key = (_qa, _sa)
            if _key not in _best or _row["bitscore"] > _best[_key][0]:
                _qcov = _row["aln_len"] / _qlen[_qa]
                _best[_key] = (_row["bitscore"], _row["pident"], _qcov)
        _ident = np.full((_n, _n), np.nan)
        _cov = np.full((_n, _n), np.nan)
        for (_qa, _sa), (_bs, _pid, _qc) in _best.items():
            _i, _j = _idx[_qa], _idx[_sa]
            _ident[_i, _j] = _pid
            _cov[_i, _j] = _qc
        np.fill_diagonal(_ident, 100.0)
        np.fill_diagonal(_cov, 1.0)
        _ident = np.fmin(_ident, _ident.T)
        _cov = np.fmin(_cov, _cov.T)
        return _accs, _ident, _cov

    _db_log = _build_nt_db()
    if not PHAGE_IDENTITY_TSV.exists() or force_rebuild.value:
        _r = subprocess.run(
            ["blastn", "-task", "megablast", "-db", str(PHAGE_BLAST_DB),
             "-query", str(PHAGES_FNA),
             "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend "
                        "sstart send evalue bitscore qlen slen",
             "-evalue", "1e-5", "-max_target_seqs", "100000", "-max_hsps", "1",
             "-num_threads", "4", "-out", str(PHAGE_IDENTITY_TSV)],
            capture_output=True, text=True)
        _db_log += f" · all-vs-all blastn rc={_r.returncode}"
    accs_phage, IDENT, COV = _matrices()
    log_identity = mo.md(f"Pairwise **% identity / coverage** for **{len(accs_phage)} phages** — {_db_log}")
    return COV, IDENT, accs_phage, log_identity


@app.cell
def _phage_dedup_controls(mo):
    ph_id_slider = mo.ui.slider(80, 100, step=1, value=95,
                                label="Duplicate % identity ≥", show_value=True)
    ph_cov_slider = mo.ui.slider(0.5, 1.0, step=0.05, value=0.85,
                                 label="Duplicate coverage ≥", show_value=True)
    mo.hstack([ph_id_slider, ph_cov_slider], justify="start", gap=2)
    return ph_cov_slider, ph_id_slider


@app.cell
def _phage_clusters(
    COV,
    IDENT,
    accs_phage,
    collections,
    mo,
    np,
    ph_cov_slider,
    ph_id_slider,
    phage_meta_df,
    pl,
):

    _meta_rows = {_r["accession"]: _r for _r in phage_meta_df.iter_rows(named=True)}

    def _uf(_k):
        return list(range(_k))
    def _find(_p, _x):
        while _p[_x] != _x:
            _p[_x] = _p[_p[_x]]; _x = _p[_x]
        return _x
    def _union(_p, _a, _b):
        _ra, _rb = _find(_p, _a), _find(_p, _b)
        if _ra != _rb:
            _p[_rb] = _ra

    def _rep_key(_i):
        _a = accs_phage[_i]; _r = _meta_rows.get(_a, {})
        return (0 if _r.get("source") == "RefSeq" else 1,
                0 if (_r.get("n_records") or 0) == 1 else 1,
                -(_r.get("length") or 0))
    def _reason(_i):
        _a = accs_phage[_i]; _r = _meta_rows.get(_a, {})
        return ", ".join([_r.get("source") or "?",
                          f"{_r.get('n_records')} record(s)",
                          f"{(_r.get('length') or 0):,} bp"])

    _n = len(accs_phage)
    _par = _uf(_n)
    for _i in range(_n):
        for _j in range(_i + 1, _n):
            if np.isnan(IDENT[_i, _j]) or np.isnan(COV[_i, _j]):
                continue
            if IDENT[_i, _j] >= ph_id_slider.value and COV[_i, _j] >= ph_cov_slider.value:
                _union(_par, _i, _j)
    _groups = collections.defaultdict(list)
    for _i in range(_n):
        _groups[_find(_par, _i)].append(_i)

    _reps = set(); _rows = []; _gid = 0
    _grp_of = {}
    for _root, _members in sorted(_groups.items()):
        _rep = min(_members, key=_rep_key)
        _reps.add(accs_phage[_rep])
        if len(_members) >= 2:
            _gid += 1
            for _i in _members:
                _grp_of[_i] = _gid
            _rows.append({
                "group": _gid, "n": len(_members),
                "representative": accs_phage[_rep], "why_rep": _reason(_rep),
                "members": ",".join(accs_phage[_i] for _i in sorted(_members)),
                "min_identity": round(min(IDENT[_i, _j] for _i in _members for _j in _members
                                          if _i < _j and not np.isnan(IDENT[_i, _j])), 1),
                "min_coverage": round(min(COV[_i, _j] for _i in _members for _j in _members
                                          if _i < _j and not np.isnan(COV[_i, _j])), 2),
            })
    dup_groups_df = pl.DataFrame(_rows) if _rows else pl.DataFrame(
        schema={"group": pl.Int64, "n": pl.Int64, "representative": pl.Utf8,
                "why_rep": pl.Utf8, "members": pl.Utf8,
                "min_identity": pl.Float64, "min_coverage": pl.Float64})
    membership = pl.DataFrame([{
        "accession": accs_phage[_i],
        "dup_group": _grp_of.get(_i),
        "is_rep": accs_phage[_i] in _reps,
    } for _i in range(_n)])

    phage_rep_set = _reps
    sum_dedup = mo.md(
        f"**{_n} phages → {len(_reps)} representatives** ({dup_groups_df.height} duplicate "
        f"group(s) collapsed at ≥{ph_id_slider.value}% id / ≥{ph_cov_slider.value} cov).")
    mo.ui.table(dup_groups_df, selection=None, pagination=True, page_size=12)
    return phage_rep_set, sum_dedup


@app.cell
def _phage_identity_heatmap(
    IDENT,
    accs_phage,
    matrix_heatmap_plotly,
    mo,
    np,
    phage_meta_df,
    pl,
):

    _n = len(accs_phage)
    _M = IDENT.astype(float)
    _vmin = float(np.nanmin(_M))
    _tk = [a for a in accs_phage]
    _side = [(phage_meta_df.filter(pl.col("accession") == a)["source"][0]
              if phage_meta_df.filter(pl.col("accession") == a).height else "?")
             for a in accs_phage]
    _fig = matrix_heatmap_plotly(
        _M, _tk, _vmin, 100.0, f"% identity<br>({_vmin:.1f}–100)",
        f"Phage genome % identity — {_n} phages",
        side_groups=_side, height=680, width=760, colorscale="YlOrRd")
    log_heat = mo.md(f"**Phage % identity** over **{_n} phages** (strips = RefSeq/GenBank).")
    _fig
    return


@app.cell(hide_code=True)
def _step5_md(mo):
    mo.md(r"""
    ## Step 5 — Compile a protein database for the deduplicated phages

    Phage genome downloads carry **no NCBI annotation**, so coding sequences are
    predicted with **pyrodigal** (meta mode) on each representative phage. This writes
    `db/proteins.faa` (HMMER / PSI-BLAST target) and `db/genes.tsv` (CDS coordinates for
    co-location in Step 8), then builds a BLAST protein DB.
    """)
    return


@app.cell
def _phage_proteins(
    GENES_TSV,
    PHAGES_DIR,
    PROTEIN_BLAST_DB,
    PROTEIN_FAA,
    Path,
    SeqIO,
    force_rebuild,
    mo,
    phage_rep_set,
    subprocess,
):

    def _build_proteins():
        _log = []
        if PROTEIN_FAA.exists() and GENES_TSV.exists() and not force_rebuild.value:
            _log.append("proteins.faa cached")
        else:
            import pyrodigal
            _gene_rows = []
            with open(PROTEIN_FAA, "w") as _out:
                for _acc in sorted(phage_rep_set):
                    _fna = PHAGES_DIR / f"{_acc}.fna"
                    if not _fna.exists():
                        continue
                    _seq = "".join(_l.strip() for _l in _fna.read_text().splitlines()
                                   if not _l.startswith(">"))
                    _finder = pyrodigal.GeneFinder(meta=True)
                    for _i, _gene in enumerate(_finder.find_genes(_seq.encode()), 1):
                        _pid = f"{_acc}_PRED_{_i}"
                        _out.write(f">{_acc}|{_pid} predicted protein\n{_gene.translate()}\n")
                        _gene_rows.append({
                            "genome": _acc, "protein": _pid, "contig": _acc,
                            "start": _gene.begin, "end": _gene.end,
                            "strand": "+" if _gene.strand > 0 else "-",
                        })
            with open(GENES_TSV, "w") as _fh:
                _fh.write("genome\tprotein\tcontig\tstart\tend\tstrand\n")
                for _g in _gene_rows:
                    _fh.write(f"{_g['genome']}\t{_g['protein']}\t{_g['contig']}\t"
                              f"{_g['start']}\t{_g['end']}\t{_g['strand']}\n")
            _log.append(f"{len(_gene_rows)} CDS predicted for {len(phage_rep_set)} phages → proteins.faa")
        return _log

    def _build_blastdb(_log):
        _idx = Path(str(PROTEIN_BLAST_DB) + ".pin")
        if _idx.exists() and not force_rebuild.value:
            _log.append("BLAST DB cached")
            return
        if not PROTEIN_FAA.exists():
            _log.append("⚠️ proteins.faa missing")
            return
        _r = subprocess.run(["makeblastdb", "-in", str(PROTEIN_FAA), "-dbtype", "prot",
                             "-out", str(PROTEIN_BLAST_DB)], capture_output=True, text=True)
        _log.append(f"makeblastdb rc={_r.returncode}")

    _log = _build_proteins()
    _build_blastdb(_log)
    n_prot = 0
    if PROTEIN_FAA.exists():
        for _rec in SeqIO.parse(str(PROTEIN_FAA), "fasta"):
            n_prot += 1
    log_proteins = mo.md(f"**Proteins** — {'; '.join(_log)} · **{n_prot} proteins** total.")
    return log_proteins, n_prot


@app.cell(hide_code=True)
def _step6_md(mo):
    mo.md(r"""
    ## Step 6 — Search for recombinases and exonucleases (HMMER + PSI-BLAST)

    Identical machinery to the main workflow, now run against the **deduplicated phage
    protein database**: six Pfam profiles (RecT, ERF, Rad52_Rad22, SSAP_Sak, YqaJ, Gam)
    via **hmmsearch**, plus **PSI-BLAST** with the SSAP seeds (Redβ P03698, RecT P33228,
    Pluβ Q7N2Y8) and exonuclease seeds (λExo P03697, RecE P15032). The merged catalog
    tags each protein with a role (SSAP / EXO / accessory) and method.
    """)
    return


@app.cell
def _db_hmms_seeds(HMM_DIR, QUERY_DIR, force_rebuild, gzip, mo, subprocess):

    FAMILY_TABLE = {
        "PF03837": ("RecT", "SSAP"),
        "PF04404": ("ERF", "SSAP"),
        "PF04098": ("Rad52_Rad22", "SSAP"),
        "PF06378": ("SSAP_Sak", "SSAP"),
        "PF09588": ("YqaJ", "EXO"),
        "PF06064": ("Gam", "accessory"),
    }
    SSAP_SEEDS = {"P03698": "Red_beta", "P33228": "RecT_Ecoli", "Q7N2Y8": "Plu_beta"}
    EXO_SEEDS = {"P03697": "lambda_exo", "P15032": "RecE"}

    def _fetch_hmms(_log):
        _need = [pf for pf in FAMILY_TABLE if force_rebuild.value or not (HMM_DIR / f"{pf}.hmm").exists()]
        if not _need:
            _log.append("all Pfam HMMs cached")
            return
        for pf in _need:
            _url = f"https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam/{pf}/?annotation=hmm"
            _r = subprocess.run(["curl", "-sL", _url], capture_output=True)
            _data = _r.stdout or b""
            if _data[:2] == b"\x1f\x8b":
                try:
                    _data = gzip.decompress(_data)
                except OSError:
                    pass
            if _data.startswith(b"HMMER3"):
                (HMM_DIR / f"{pf}.hmm").write_bytes(_data)
                _log.append(f"{pf} ✅")
            else:
                _log.append(f"{pf} ⚠️ fetch failed")

    def _fetch_seeds(_log):
        _groups = {"all_3_ssap_queries.faa": SSAP_SEEDS, "exo_seeds.faa": EXO_SEEDS}
        for _fname, _accs in _groups.items():
            _dst = QUERY_DIR / _fname
            if _dst.exists() and not force_rebuild.value:
                continue
            _recs = []
            for _acc in _accs:
                _r = subprocess.run(["curl", "-sL",
                                     f"https://rest.uniprot.org/uniprotkb/{_acc}.fasta"],
                                    capture_output=True, text=True)
                if _r.stdout.startswith(">"):
                    _recs.append(_r.stdout.strip())
            if _recs:
                _dst.write_text("\n".join(_recs) + "\n")
                _log.append(f"{_fname}: {len(_recs)} seed(s)")

    _db_log = []
    _fetch_hmms(_db_log)
    _fetch_seeds(_db_log)
    log_refs = mo.md("**Reference data** — " + ("; ".join(_db_log) if _db_log else "cached"))
    return (FAMILY_TABLE,)


@app.cell
def _combine_role_hits(collections, mo, pl):

    def combine_role_hits(hmm_df, psi_df, families, psi_method):
        _agg = collections.defaultdict(
            lambda: {"hmm_families": set(), "hmm_evalue": None,
                     "psi_seeds": set(), "psi_evalue": None})
        if hmm_df is not None and hmm_df.height:
            for _r in hmm_df.filter(pl.col("family").is_in(list(families))).iter_rows(named=True):
                _k = (_r["genome"], _r["protein"])
                _agg[_k]["hmm_families"].add(_r["family"])
                _e = _r["i_evalue"]
                _agg[_k]["hmm_evalue"] = _e if _agg[_k]["hmm_evalue"] is None else min(_agg[_k]["hmm_evalue"], _e)
        if psi_df is not None and psi_df.height:
            for _r in psi_df.iter_rows(named=True):
                _k = (_r["genome"], _r["protein"])
                _agg[_k]["psi_seeds"].add(_r["seed"])
                _e = _r["evalue"]
                _agg[_k]["psi_evalue"] = _e if _agg[_k]["psi_evalue"] is None else min(_agg[_k]["psi_evalue"], _e)
        _rows = []
        for (_genome, _protein), _d in _agg.items():
            _methods = []
            if _d["hmm_families"]:
                _methods.append("HMM")
            if _d["psi_seeds"]:
                _methods.append(psi_method)
            _evs = [x for x in (_d["hmm_evalue"], _d["psi_evalue"]) if x is not None]
            _rows.append({
                "genome": _genome, "protein": _protein,
                "hmm_families": ", ".join(sorted(_d["hmm_families"])) or "—",
                "psiblast_seeds": ", ".join(sorted(_d["psi_seeds"])) or "—",
                "method": "+".join(_methods),
                "best_evalue": min(_evs) if _evs else None,
                "hmm_evalue": _d["hmm_evalue"], "psi_evalue": _d["psi_evalue"],
            })
        if not _rows:
            return pl.DataFrame(schema={"genome": pl.Utf8, "protein": pl.Utf8,
                                        "hmm_families": pl.Utf8, "psiblast_seeds": pl.Utf8,
                                        "method": pl.Utf8, "best_evalue": pl.Float64,
                                        "hmm_evalue": pl.Float64, "psi_evalue": pl.Float64})
        return pl.DataFrame(_rows).sort("best_evalue")

    mo.md("*`combine_role_hits` merges HMMER and PSI-BLAST evidence per protein.*")
    return (combine_role_hits,)


@app.cell
def _hmm_config(FAMILY_TABLE, mo):

    HMM_FAMILIES = {pf: name for pf, (name, role) in FAMILY_TABLE.items()}
    SSAP_FAMILIES = {name for pf, (name, role) in FAMILY_TABLE.items() if role == "SSAP"}
    EXO_FAMILIES = {name for pf, (name, role) in FAMILY_TABLE.items() if role == "EXO"}
    ACCESSORY_FAMILIES = {name for pf, (name, role) in FAMILY_TABLE.items() if role == "accessory"}

    hmm_run_button = mo.ui.run_button(label="⟳ Re-run hmmsearch (6 Pfam profiles)")
    hmm_evalue_exp = mo.ui.slider(start=-50, stop=1, step=1, value=-5,
                                  label="Called-hit i-E-value ≤ 10^x", show_value=True)
    mo.hstack([hmm_evalue_exp, hmm_run_button], justify="start", gap=2)
    return (
        ACCESSORY_FAMILIES,
        EXO_FAMILIES,
        HMM_FAMILIES,
        SSAP_FAMILIES,
        hmm_evalue_exp,
        hmm_run_button,
    )


@app.cell
def _hmm_load(
    HMM_DIR,
    HMM_FAMILIES,
    PROTEIN_FAA,
    RESULTS_DIR,
    hmm_run_button,
    mo,
    n_prot,
    phage_rep_set,
    pl,
    subprocess,
):

    def _parse_domtbl(_path, _fam):
        _out = []
        with open(_path) as _fh:
            for _line in _fh:
                if _line.startswith("#"):
                    continue
                _c = _line.split()
                if len(_c) < 23:
                    continue
                _genome, _, _protein = _c[0].partition("|")
                _out.append({
                    "family": _fam, "genome": _genome, "protein": _protein or _c[0],
                    "i_evalue": float(_c[12]), "dom_score": float(_c[13]),
                    "tlen": int(_c[2]),
                    "ali_from": int(_c[17]), "ali_to": int(_c[18]),
                    "desc": " ".join(_c[22:]),
                })
        return _out

    def _load_hmm(_do_rerun):
        _need = _do_rerun or not all((RESULTS_DIR / f"hmmsearch_{_pf}.domtbl").exists() for _pf in HMM_FAMILIES)
        if _need and PROTEIN_FAA.exists() and HMM_DIR.exists():
            for _pf in HMM_FAMILIES:
                subprocess.run(
                    ["hmmsearch", "-o", str(RESULTS_DIR / f"hmmsearch_{_pf}.out"),
                     "--domtblout", str(RESULTS_DIR / f"hmmsearch_{_pf}.domtbl"),
                     str(HMM_DIR / f"{_pf}.hmm"), str(PROTEIN_FAA)],
                    capture_output=True, text=True)
        _rows = []
        for _pf, _fam in HMM_FAMILIES.items():
            _p = RESULTS_DIR / f"hmmsearch_{_pf}.domtbl"
            if _p.exists():
                _rows += _parse_domtbl(_p, _fam)
        schema = {"family": pl.Utf8, "genome": pl.Utf8, "protein": pl.Utf8,
                  "i_evalue": pl.Float64, "dom_score": pl.Float64, "tlen": pl.Int64,
                  "ali_from": pl.Int64, "ali_to": pl.Int64, "desc": pl.Utf8}
        _df = pl.DataFrame(_rows, schema=schema) if _rows else pl.DataFrame(schema=schema)
        if _df.height:
            _df = _df.filter(pl.col("genome").is_in(list(phage_rep_set)))
        return _df

    hmm_hits_all = _load_hmm(hmm_run_button.value)
    log_hmm_load = mo.md(f"Parsed **{hmm_hits_all.height} domain rows** across "
                         f"{hmm_hits_all['genome'].n_unique() if hmm_hits_all.height else 0} phages "
                         f"({n_prot} proteins searched).")
    mo.vstack([log_hmm_load])
    return (hmm_hits_all,)


@app.cell
def _hmm_filter(hmm_evalue_exp, hmm_hits_all, mo, pl):
    _thr = 10.0 ** hmm_evalue_exp.value
    if hmm_hits_all.height:
        hmm_hits_df = (hmm_hits_all.filter(pl.col("i_evalue") <= _thr)
                       .sort("i_evalue")
                       .unique(subset=["genome", "protein", "family"], keep="first")
                       .sort(["family", "i_evalue"]))
    else:
        hmm_hits_df = hmm_hits_all
    log_hmm = mo.md(
        f"**{hmm_hits_df.height} called profile hits** at i-E ≤ {_thr:.0e} across "
        f"{hmm_hits_df['genome'].n_unique() if hmm_hits_df.height else 0} phages.")
    return (hmm_hits_df,)


@app.cell
def _(SSAP_FAMILIES, hmm_evalue_exp, hmm_hits_df, mo, phage_meta_df, pl):
    _hmm_rec = (hmm_hits_df.filter(pl.col("family").is_in(list(SSAP_FAMILIES)))
                .join(phage_meta_df.select([pl.col("accession").alias("genome"), "name"]),
                      on="genome", how="left")
                .select(["name", "genome", "protein", "family", "i_evalue", "dom_score",
                         "ali_from", "ali_to", "desc"])
                .sort("i_evalue"))
    _thr = 10.0 ** hmm_evalue_exp.value
    sum_hmm_rec = mo.md(
        f"**{_hmm_rec.height} recombinase hits** from HMMER at i-E ≤ {_thr:.0e} · "
        f"{_hmm_rec['protein'].n_unique() if _hmm_rec.height else 0} unique proteins across "
        f"{_hmm_rec['genome'].n_unique() if _hmm_rec.height else 0} phages. "
        f"Profile families: {', '.join(sorted(SSAP_FAMILIES))}.")
    mo.ui.table(_hmm_rec, selection=None, pagination=True, page_size=15)
    return (sum_hmm_rec,)


@app.cell
def _psi_ssap_config(mo):
    psi_evalue_exp = mo.ui.slider(start=-50, stop=1, step=1, value=-3,
                                  label="Keep SSAP hits with E ≤ 10^x", show_value=True)
    psi_run_button = mo.ui.run_button(label="⟳ Re-run PSI-BLAST (3 SSAP seeds)")
    mo.hstack([psi_evalue_exp, psi_run_button], justify="start", gap=2)
    return psi_evalue_exp, psi_run_button


@app.cell
def _psi_ssap_load(
    PROTEIN_BLAST_DB,
    QUERY_DIR,
    RESULTS_DIR,
    mo,
    n_prot,
    parse_psiblast_tsv,
    phage_rep_set,
    pl,
    psi_run_button,
    subprocess,
):

    _seed_faa = QUERY_DIR / "all_3_ssap_queries.faa"
    _psiblast_tsv = RESULTS_DIR / "psiblast_ssap.tsv"

    def _load(_do_rerun):
        if (_do_rerun or not _psiblast_tsv.exists()) and _seed_faa.exists():
            subprocess.run(
                ["psiblast", "-query", str(_seed_faa), "-db", str(PROTEIN_BLAST_DB),
                 "-num_iterations", "3", "-evalue", "1e-3", "-inclusion_ethresh", "1e-3",
                 "-outfmt", "6 std qcovhsp", "-num_threads", "4", "-out", str(_psiblast_tsv)],
                capture_output=True, text=True)
        schema = {"seed": pl.Utf8, "genome": pl.Utf8, "protein": pl.Utf8,
                  "pident": pl.Float64, "aln_len": pl.Int64, "evalue": pl.Float64,
                  "bitscore": pl.Float64, "qcovhsp": pl.Int64}
        _rows = parse_psiblast_tsv(_psiblast_tsv) if _psiblast_tsv.exists() else []
        _df = pl.DataFrame(_rows, schema=schema) if _rows else pl.DataFrame(schema=schema)
        if _df.height:
            _df = _df.filter(pl.col("genome").is_in(list(phage_rep_set)))
        return _df

    psi_hits_all = _load(psi_run_button.value)
    mo.md(f"Parsed **{psi_hits_all.height} SSAP PSI-BLAST HSP rows** ({n_prot} proteins).")
    return (psi_hits_all,)


@app.cell
def _psi_exo_config(mo):
    psi_exo_evalue_exp = mo.ui.slider(start=-50, stop=1, step=1, value=-3,
                                      label="Keep EXO hits with E ≤ 10^x", show_value=True)
    psi_exo_run_button = mo.ui.run_button(label="⟳ Re-run PSI-BLAST (exonuclease seeds)")
    mo.hstack([psi_exo_evalue_exp, psi_exo_run_button], justify="start", gap=2)
    return psi_exo_evalue_exp, psi_exo_run_button


@app.cell
def _psi_exo_load(
    PROTEIN_BLAST_DB,
    QUERY_DIR,
    RESULTS_DIR,
    mo,
    n_prot,
    parse_psiblast_tsv,
    phage_rep_set,
    pl,
    psi_exo_run_button,
    subprocess,
):

    _exo_seed_faa = QUERY_DIR / "exo_seeds.faa"
    _psiblast_exo_tsv = RESULTS_DIR / "psiblast_exo.tsv"

    def _load(_do_rerun):
        if (_do_rerun or not _psiblast_exo_tsv.exists()) and _exo_seed_faa.exists():
            subprocess.run(
                ["psiblast", "-query", str(_exo_seed_faa), "-db", str(PROTEIN_BLAST_DB),
                 "-num_iterations", "3", "-evalue", "1e-3", "-inclusion_ethresh", "1e-3",
                 "-outfmt", "6 std qcovhsp", "-num_threads", "4", "-out", str(_psiblast_exo_tsv)],
                capture_output=True, text=True)
        schema = {"seed": pl.Utf8, "genome": pl.Utf8, "protein": pl.Utf8,
                  "pident": pl.Float64, "aln_len": pl.Int64, "evalue": pl.Float64,
                  "bitscore": pl.Float64, "qcovhsp": pl.Int64}
        _rows = parse_psiblast_tsv(_psiblast_exo_tsv) if _psiblast_exo_tsv.exists() else []
        _df = pl.DataFrame(_rows, schema=schema) if _rows else pl.DataFrame(schema=schema)
        if _df.height:
            _df = _df.filter(pl.col("genome").is_in(list(phage_rep_set)))
        return _df

    psi_exo_hits_all = _load(psi_exo_run_button.value)
    mo.md(f"Parsed **{psi_exo_hits_all.height} exonuclease PSI-BLAST HSP rows** ({n_prot} proteins).")
    return (psi_exo_hits_all,)


@app.cell
def _psi_filter(
    mo,
    pl,
    psi_evalue_exp,
    psi_exo_evalue_exp,
    psi_exo_hits_all,
    psi_hits_all,
):
    _thr = 10.0 ** psi_evalue_exp.value
    if psi_hits_all.height:
        psi_hits_df = (psi_hits_all.filter(pl.col("evalue") <= _thr)
                       .sort("evalue").unique(subset=["seed", "genome", "protein"], keep="first"))
    else:
        psi_hits_df = psi_hits_all
    _thr_exo = 10.0 ** psi_exo_evalue_exp.value
    if psi_exo_hits_all.height:
        psi_exo_hits_df = (psi_exo_hits_all.filter(pl.col("evalue") <= _thr_exo)
                           .sort("evalue").unique(subset=["seed", "genome", "protein"], keep="first"))
    else:
        psi_exo_hits_df = psi_exo_hits_all
    log_psi = mo.md(
        f"**{psi_hits_df.height} SSAP** + **{psi_exo_hits_df.height} exonuclease** PSI-BLAST hits.")
    return psi_exo_hits_df, psi_hits_df


@app.cell
def _(mo, phage_meta_df, pl, psi_evalue_exp, psi_hits_df):
    _psi_rec = (psi_hits_df
                .join(phage_meta_df.select([pl.col("accession").alias("genome"), "name"]),
                      on="genome", how="left")
                .select(["name", "genome", "protein", "seed", "pident", "aln_len",
                         "qcovhsp", "evalue", "bitscore"])
                .sort("evalue"))
    _thr = 10.0 ** psi_evalue_exp.value
    sum_psi_rec = mo.md(
        f"**{_psi_rec.height} recombinase hits** from PSI-BLAST at E ≤ {_thr:.0e} · "
        f"{_psi_rec['protein'].n_unique() if _psi_rec.height else 0} unique proteins across "
        f"{_psi_rec['genome'].n_unique() if _psi_rec.height else 0} phages.")
    mo.ui.table(_psi_rec, selection=None, pagination=True, page_size=15)
    return (sum_psi_rec,)


@app.cell
def _(SSAP_FAMILIES, combine_role_hits, hmm_hits_df, mo, pl, psi_hits_df):
    recombine_hits = combine_role_hits(hmm_hits_df, psi_hits_df, SSAP_FAMILIES, "SSAP-PSI")
    _n_hmm = recombine_hits.filter(pl.col("method").str.contains("HMM")).height if recombine_hits.height else 0
    _n_psi = recombine_hits.filter(pl.col("method").str.contains("PSI")).height if recombine_hits.height else 0
    sum_merged_rec = mo.md(
        f"**{recombine_hits.height} recombinase hits** merging HMMER and PSI-BLAST "
        f"({_n_hmm} with profile evidence, {_n_psi} with PSI-BLAST evidence) across "
        f"{recombine_hits['genome'].n_unique() if recombine_hits.height else 0} phages.")
    mo.ui.table(recombine_hits, selection=None, pagination=True, page_size=15)
    return recombine_hits, sum_merged_rec


@app.cell
def _(
    SSAP_FAMILIES,
    alt,
    hmm_hits_df,
    mo,
    pdf,
    pl,
    psi_hits_df,
    recombine_hits,
    result,
):
    if recombine_hits.height:
        _hmm_c = (hmm_hits_df.filter(pl.col("family").is_in(list(SSAP_FAMILIES)))
                  .group_by("family").len().with_columns(pl.lit("HMM").alias("method")))
        _psi_c = (psi_hits_df.group_by("seed").len().rename({"seed": "family"})
                  .with_columns(pl.lit("PSI-BLAST").alias("method")))
        _counts = pl.concat([_hmm_c, _psi_c], how="vertical")
        _chart = (alt.Chart(pdf(_counts)).mark_bar().encode(
            x=alt.X("len:Q", title="hits"),
            y=alt.Y("family:N", sort="-x", title="Pfam / PSI-BLAST family"),
            color=alt.Color("method:N", title="search",
                            scale=alt.Scale(range=["#4c78a8", "#f58518"])),
            tooltip=["family", "method", "len"]).properties(
            height=280, width=460, title="Recombinase hits per Pfam / PSI-BLAST family"))
        _out = mo.ui.altair_chart(_chart)
    else:
        _out = mo.md("*No recombinase hits at the current thresholds.*")
    result(_out)
    return


@app.cell
def _(alt, collections, mo, pdf, pl, recombine_hits, result):
    if recombine_hits.height:
        _cnt = collections.Counter(
            (fam.strip(), meth)
            for fams, meth in zip(recombine_hits["hmm_families"].to_list(),
                                  recombine_hits["method"].to_list())
            for fam in (fams.split(", ") if fams and fams != "—"
                        else ["PSI-BLAST only (no Pfam)"]))
        _rows = [{"superfamily": k[0], "method": k[1], "hits": v} for k, v in _cnt.most_common()]
        _sftot = collections.Counter()
        for _r in _rows:
            _sftot[_r["superfamily"]] += _r["hits"]
        _chart = (alt.Chart(pdf(pl.DataFrame(_rows)))
                  .mark_bar()
                  .encode(
                      x=alt.X("hits:Q", title="merged hits"),
                      y=alt.Y("superfamily:N", sort="-x", title="Pfam superfamily"),
                      color=alt.Color("method:N", title="evidence",
                                      scale=alt.Scale(
                                          domain=["HMM", "SSAP-PSI", "HMM+SSAP-PSI"],
                                          range=["#4c78a8", "#f58518", "#54a24b"])),
                      tooltip=["superfamily", "method", "hits"])
                  .properties(height=max(140, 30 * len(_sftot)), width=460,
                              title="Merged recombinase hits by superfamily (Pfam)"))
        _out = mo.ui.altair_chart(_chart)
        sum_superfam = mo.md(
            f"**{recombine_hits.height} merged recombinase hits** across "
            f"**{len(_sftot)} superfamilies** — "
            + " · ".join(f"{k} {v}" for k, v in _sftot.most_common())
            + ". Hits found only by PSI-BLAST, with no Pfam profile, appear as a "
              "`PSI-BLAST only (no Pfam)` bucket.")
    else:
        sum_superfam = mo.md("*No merged recombinase hits at the current thresholds.*")
        _out = sum_superfam
    result(_out)
    return (sum_superfam,)


@app.cell
def _hmm_exo_filter(
    ACCESSORY_FAMILIES,
    EXO_FAMILIES,
    hmm_evalue_exp,
    hmm_hits_all,
    pl,
):
    _thr = 10.0 ** hmm_evalue_exp.value
    if hmm_hits_all.height:
        hmm_exo_hits_df = (hmm_hits_all.filter(pl.col("i_evalue") <= _thr)
                           .filter(pl.col("family").is_in(list(EXO_FAMILIES | ACCESSORY_FAMILIES)))
                           .sort("i_evalue")
                           .unique(subset=["genome", "protein", "family"], keep="first"))
    else:
        hmm_exo_hits_df = hmm_hits_all
    return (hmm_exo_hits_df,)


@app.cell
def _(hmm_evalue_exp, hmm_exo_hits_df, mo, phage_meta_df, pl):
    _hmm_exo = (hmm_exo_hits_df
                .join(phage_meta_df.select([pl.col("accession").alias("genome"), "name"]),
                      on="genome", how="left")
                .select(["name", "genome", "protein", "family", "i_evalue", "dom_score",
                         "ali_from", "ali_to", "desc"])
                .sort("i_evalue"))
    _thr = 10.0 ** hmm_evalue_exp.value
    sum_hmm_exo = mo.md(
        f"**{_hmm_exo.height} exonuclease / inhibitor hits** from HMMER (YqaJ, Gam) at "
        f"i-E ≤ {_thr:.0e} · {_hmm_exo['genome'].n_unique() if _hmm_exo.height else 0} phages.")
    mo.ui.table(_hmm_exo, selection=None, pagination=True, page_size=15)
    return (sum_hmm_exo,)


@app.cell
def _(mo, phage_meta_df, pl, psi_exo_evalue_exp, psi_exo_hits_df):
    _psi_exo = (psi_exo_hits_df
                .join(phage_meta_df.select([pl.col("accession").alias("genome"), "name"]),
                      on="genome", how="left")
                .select(["name", "genome", "protein", "seed", "pident", "aln_len",
                         "qcovhsp", "evalue", "bitscore"])
                .sort("evalue"))
    _thr = 10.0 ** psi_exo_evalue_exp.value
    sum_psi_exo = mo.md(
        f"**{_psi_exo.height} exonuclease hits** from PSI-BLAST at E ≤ {_thr:.0e} · "
        f"{_psi_exo['protein'].n_unique() if _psi_exo.height else 0} unique proteins across "
        f"{_psi_exo['genome'].n_unique() if _psi_exo.height else 0} phages.")
    mo.ui.table(_psi_exo, selection=None, pagination=True, page_size=15)
    return (sum_psi_exo,)


@app.cell
def _(
    ACCESSORY_FAMILIES,
    EXO_FAMILIES,
    combine_role_hits,
    hmm_exo_hits_df,
    mo,
    psi_exo_hits_df,
):
    exo_hits = combine_role_hits(hmm_exo_hits_df, psi_exo_hits_df, EXO_FAMILIES | ACCESSORY_FAMILIES, "EXO-PSI")
    sum_merged_exo = mo.md(
        f"**{exo_hits.height} exonuclease / host-nuclease inhibitor hits** merging HMMER "
        f"(YqaJ, Gam) and PSI-BLAST across {exo_hits['genome'].n_unique() if exo_hits.height else 0} phages.")
    mo.ui.table(exo_hits, selection=None, pagination=True, page_size=15)
    return exo_hits, sum_merged_exo


@app.cell
def _(alt, exo_hits, hmm_exo_hits_df, mo, pdf, pl, psi_exo_hits_df, result):
    if exo_hits.height:
        _hmm_c = (hmm_exo_hits_df.group_by("family").len().with_columns(pl.lit("HMM").alias("method")))
        _psi_c = (psi_exo_hits_df.group_by("seed").len().rename({"seed": "family"})
                  .with_columns(pl.lit("PSI-BLAST").alias("method")))
        _counts = pl.concat([_hmm_c, _psi_c], how="vertical")
        _chart = (alt.Chart(pdf(_counts)).mark_bar().encode(
            x=alt.X("len:Q", title="hits"),
            y=alt.Y("family:N", sort="-x", title="Pfam / PSI-BLAST family"),
            color=alt.Color("method:N", title="search",
                            scale=alt.Scale(range=["#e45756", "#72b7b2"])),
            tooltip=["family", "method", "len"]).properties(
            height=260, width=460, title="Exonuclease / inhibitor hits per Pfam / PSI-BLAST family"))
        _out = mo.ui.altair_chart(_chart)
    else:
        _out = mo.md("*No exonuclease hits at the current thresholds.*")
    result(_out)
    return


@app.cell
def _(alt, collections, exo_hits, mo, pdf, pl, result):
    if exo_hits.height:
        _cnt = collections.Counter(
            (fam.strip(), meth)
            for fams, meth in zip(exo_hits["hmm_families"].to_list(),
                                  exo_hits["method"].to_list())
            for fam in (fams.split(", ") if fams and fams != "—"
                        else ["PSI-BLAST only (no Pfam)"]))
        _rows = [{"superfamily": k[0], "method": k[1], "hits": v} for k, v in _cnt.most_common()]
        _sftot = collections.Counter()
        for _r in _rows:
            _sftot[_r["superfamily"]] += _r["hits"]
        _chart = (alt.Chart(pdf(pl.DataFrame(_rows)))
                  .mark_bar()
                  .encode(
                      x=alt.X("hits:Q", title="merged hits"),
                      y=alt.Y("superfamily:N", sort="-x", title="Pfam superfamily"),
                      color=alt.Color("method:N", title="evidence",
                                      scale=alt.Scale(
                                          domain=["HMM", "EXO-PSI", "HMM+EXO-PSI"],
                                          range=["#e45756", "#72b7b2", "#54a24b"])),
                      tooltip=["superfamily", "method", "hits"])
                  .properties(height=max(120, 30 * len(_sftot)), width=460,
                              title="Merged exonuclease / inhibitor hits by superfamily (Pfam)"))
        _out = mo.ui.altair_chart(_chart)
        sum_superfam_exo = mo.md(
            f"**{exo_hits.height} merged exonuclease / inhibitor hits** across "
            f"**{len(_sftot)} superfamilies** — "
            + " · ".join(f"{k} {v}" for k, v in _sftot.most_common())
            + ". Hits found only by PSI-BLAST, with no Pfam profile, appear as a "
              "`PSI-BLAST only (no Pfam)` bucket.")
    else:
        sum_superfam_exo = mo.md("*No merged exonuclease hits at the current thresholds.*")
        _out = sum_superfam_exo
    result(_out)
    return (sum_superfam_exo,)


@app.cell
def _recomb_catalog(
    ACCESSORY_FAMILIES,
    EXO_FAMILIES,
    SSAP_FAMILIES,
    hmm_hits_df,
    mo,
    phage_meta_df,
    pl,
    psi_exo_hits_df,
    psi_hits_df,
):

    _seed_labels = {"P03698": "Red_beta", "P33228": "RecT_Ecoli", "Q7N2Y8": "Plu_beta",
                    "P03697": "lambda_exo", "P15032": "RecE"}

    def _agg_psi(_df, _seedcol, _evcol):
        if _df.height:
            return (_df.group_by(["genome", "protein"]).agg(
                pl.col("seed").unique().sort().alias("_s"),
                pl.col("evalue").min().alias(_evcol))
                .with_columns(pl.col("_s").list.join(",").alias(_seedcol)).drop("_s"))
        return pl.DataFrame(schema={"genome": pl.Utf8, "protein": pl.Utf8,
                                    _evcol: pl.Float64, _seedcol: pl.Utf8})

    def _build_catalog(_hmm_df, _ssap_df, _exo_df):
        _hmm_side = (_hmm_df.select(["genome", "protein", "family", "i_evalue", "desc"])
                     .rename({"i_evalue": "hmm_evalue"})) if _hmm_df.height else pl.DataFrame(
            schema={"genome": pl.Utf8, "protein": pl.Utf8, "family": pl.Utf8,
                    "hmm_evalue": pl.Float64, "desc": pl.Utf8})
        _cat = (_hmm_side
                .join(_agg_psi(_ssap_df, "ssap_seeds", "ssap_evalue"),
                      on=["genome", "protein"], how="full", coalesce=True)
                .join(_agg_psi(_exo_df, "exo_seeds", "exo_evalue"),
                      on=["genome", "protein"], how="full", coalesce=True))
        _cat = _cat.with_columns(
            pl.when(pl.col("exo_seeds").is_not_null() | pl.col("family").is_in(list(EXO_FAMILIES)))
              .then(pl.lit("EXO"))
              .when(pl.col("ssap_seeds").is_not_null() | pl.col("family").is_in(list(SSAP_FAMILIES)))
              .then(pl.lit("SSAP"))
              .when(pl.col("family").is_in(list(ACCESSORY_FAMILIES))).then(pl.lit("accessory"))
              .otherwise(pl.lit("other")).alias("role"),
            pl.concat_list([
                pl.when(pl.col("family").is_not_null()).then(pl.lit("HMM")),
                pl.when(pl.col("ssap_seeds").is_not_null()).then(pl.lit("SSAP-PSI")),
                pl.when(pl.col("exo_seeds").is_not_null()).then(pl.lit("EXO-PSI")),
            ]).list.drop_nulls().list.join("+").alias("method"),
        )

        def _family(_r):
            if _r["family"]:
                return _r["family"]
            if _r["exo_seeds"]:
                return "YqaJ(PSI-exo)"
            if _r["ssap_seeds"]:
                _s = _r["ssap_seeds"].split(",")[0]
                return f"SSAP(PSI:{_seed_labels.get(_s, _s)})"
            return "other(PSI)"

        _cat = _cat.with_columns(pl.Series("family", [_family(_r) for _r in _cat.iter_rows(named=True)], dtype=pl.Utf8))
        return _cat.sort(["role", "family", "hmm_evalue"])

    recomb_df = _build_catalog(hmm_hits_df, psi_hits_df, psi_exo_hits_df)
    _disp = recomb_df.join(phage_meta_df.select([pl.col("accession").alias("genome"), "name"]),
                           on="genome", how="left")
    _roles = ({r: int(n) for r, n in zip(
        recomb_df["role"].value_counts().sort("role")["role"],
        recomb_df["role"].value_counts().sort("role")["count"])}
        if recomb_df.height else {})
    sum_catalog = mo.md(
        f"**{recomb_df.height} recombineering proteins** across "
        f"{recomb_df['genome'].n_unique() if recomb_df.height else 0} phages · roles: {_roles}")
    mo.ui.table(_disp.select(["role", "family", "method", "name", "genome", "protein",
                              "hmm_evalue", "ssap_evalue", "exo_evalue"]),
                selection=None, pagination=True, page_size=12)
    return recomb_df, sum_catalog


@app.cell
def _recomb_seqs(PROTEIN_FAA, SeqIO, mo, recomb_df):
    def _load_seqs(_faa, _wanted_ids):
        _seqs = {}
        if not _faa.exists():
            return _seqs
        for _rec in SeqIO.parse(str(_faa), "fasta"):
            _parts = _rec.id.split("|")
            _pid = _parts[1] if len(_parts) >= 2 else _rec.id
            if _pid in _wanted_ids and _pid not in _seqs:
                _seqs[_pid] = str(_rec.seq)
        return _seqs

    recomb_seqs = _load_seqs(PROTEIN_FAA, set(recomb_df["protein"].to_list()))
    log_seqs = mo.md(f"Loaded **{len(recomb_seqs)} / {recomb_df['protein'].n_unique()}** "
                     f"recombinase protein sequences.")
    return (recomb_seqs,)


@app.cell(hide_code=True)
def _step7_md(mo):
    mo.md(r"""
    ## Step 7 — Deduplicate recombinases by % similarity

    Recombinases (SSAPs) are aligned per family with **MAFFT** and compared with three
    pairwise metrics — **% identity**, **BLOSUM62 % similarity**, and **coverage** — then
    deduplicated (single-linkage) with a tie-break to pick one representative per
    near-identical cluster, exactly as in the main workflow.
    """)
    return


@app.cell
def _uniq_matrices(
    MSA_DIR,
    SSAP_FAMILIES,
    SeqIO,
    force_rebuild,
    io,
    mo,
    np,
    pl,
    re,
    recomb_df,
    recomb_seqs,
    subprocess,
    substitution_matrices,
):

    _b62 = substitution_matrices.load("BLOSUM62")

    def _b62_score(_x, _y):
        try:
            return float(_b62[_x, _y])
        except Exception:
            return 0.0

    def _recs_for(_fam):
        _out = []
        for _r in recomb_df.filter((pl.col("family") == _fam) & (pl.col("role") == "SSAP")).iter_rows(named=True):
            _s = recomb_seqs.get(_r["protein"])
            if _s:
                _out.append((f"{_r['genome']}|{_r['protein']}", _s, _r["family"], _r["role"]))
        return _out

    def _fam_msa(_recs):
        _n = len(_recs)
        _labels = [_r[0] for _r in _recs]
        if _n == 0:
            _z = np.zeros((0, 0)); return _labels, _z, _z, _z
        if _n == 1:
            _one = np.array([[100.0]]); return _labels, _one, _one, np.array([[1.0]])
        _fasta = "\n".join(f">{i}\n{s}" for i, (_, s, _, _) in enumerate(_recs))
        _r = subprocess.run(["mafft", "--quiet", "--auto", "-"], input=_fasta,
                            capture_output=True, text=True)
        _by = {int(_a.id): str(_a.seq) for _a in SeqIO.parse(io.StringIO(_r.stdout), "fasta")}
        _lens = [len(_s) for _, _s, _, _ in _recs]
        _ident = np.full((_n, _n), np.nan); _sim = np.full((_n, _n), np.nan); _cov = np.full((_n, _n), np.nan)
        for _i in range(_n):
            for _j in range(_i, _n):
                _si, _sj = _by[_i], _by[_j]
                _both = _same = _pos = 0
                for _x, _y in zip(_si, _sj):
                    if _x != "-" and _y != "-":
                        _both += 1
                        _same += int(_x == _y)
                        _pos += int(_b62_score(_x, _y) > 0)
                _ident[_i, _j] = _ident[_j, _i] = 100.0 * _same / _both if _both else 0.0
                _sim[_i, _j] = _sim[_j, _i] = 100.0 * _pos / _both if _both else 0.0
                _m = min(_lens[_i], _lens[_j])
                _cov[_i, _j] = _cov[_j, _i] = _both / _m if _m else 0.0
        return _labels, _ident, _sim, _cov

    def _is_recombinase_family(_f):
        return (_f in SSAP_FAMILIES) or _f.startswith("SSAP(PSI:")

    def _build_all_msa():
        _d = {}
        for _fam in sorted(_f for _f in recomb_df["family"].unique().to_list()
                           if _is_recombinase_family(_f)):
            _recs = _recs_for(_fam)
            _safe = re.sub(r"[^A-Za-z0-9_.-]", "_", _fam)
            _cache = MSA_DIR / f"{_safe}.npz"
            _labels = [_r[0] for _r in _recs]
            if _cache.exists() and not force_rebuild.value:
                _z = np.load(_cache, allow_pickle=True)
                if set(map(str, _z["labels"])) == set(_labels):
                    _d[_fam] = (list(map(str, _z["labels"])), _z["ident"], _z["sim"], _z["cov"], _recs)
                    continue
            _labels, _ident, _sim, _cov = _fam_msa(_recs)
            _d[_fam] = (_labels, _ident, _sim, _cov, _recs)
            np.savez(_cache, labels=np.array(_labels), ident=_ident, sim=_sim, cov=_cov)
        return _d

    unique_msa = _build_all_msa()
    log_msa = mo.md("Per-family MAFFT alignment built for: "
          + ", ".join(f"`{f}` (n={len(v[0])})" for f, v in unique_msa.items()))
    return log_msa, unique_msa


@app.cell
def _uniq_controls(mo):
    dup_id_slider = mo.ui.slider(50, 100, step=1, value=90,
                                 label="Duplicate identity ≥ (%)", show_value=True)
    sys_id_slider = mo.ui.slider(30, 100, step=1, value=70,
                                 label="System/divergent identity ≥ (%)", show_value=True)
    cov_slider = mo.ui.slider(0.0, 1.0, step=0.05, value=0.8,
                              label="Min coverage", show_value=True)
    mo.vstack([mo.md("**Uniqueness thresholds**"), dup_id_slider, sys_id_slider, cov_slider])
    return cov_slider, dup_id_slider, sys_id_slider


@app.cell
def _uniq_clusters(
    collections,
    cov_slider,
    dup_id_slider,
    mo,
    pl,
    sys_id_slider,
    unique_msa,
):

    def single_linkage(labels, ident, cov, thr_id, thr_cov):
        _n = len(labels)
        _adj = collections.defaultdict(set)
        for _i in range(_n):
            for _j in range(_i + 1, _n):
                if ident[_i, _j] >= thr_id and cov[_i, _j] >= thr_cov:
                    _adj[_i].add(_j); _adj[_j].add(_i)
        _seen, _comps = set(), []
        for _i in range(_n):
            if _i in _seen:
                continue
            _stack, _comp = [_i], []
            while _stack:
                _x = _stack.pop()
                if _x in _seen:
                    continue
                _seen.add(_x); _comp.append(_x); _stack.extend(_adj[_x] - _seen)
            _comps.append(sorted(_comp))
        return _comps

    def rep_of(recs, idxs):
        def _key(_i):
            return (-len(recs[_i][1]), recs[_i][0])
        return min(idxs, key=_key)

    def _build_unique(_thr_dup, _thr_sys, _thr_cov):
        _rows = []
        _dup_cluster_of, _sys_cluster_of = {}, {}
        for _fam, (_labels, _ident, _sim, _cov, _recs) in unique_msa.items():
            if not _labels:
                continue
            _dup = single_linkage(_labels, _ident, _cov, _thr_dup, _thr_cov)
            _sysc = single_linkage(_labels, _ident, _cov, _thr_sys, _thr_cov)
            _sysid = {i: gi for gi, comp in enumerate(_sysc) for i in comp}
            for _ci, _comp in enumerate(_dup, 1):
                for _i in _comp:
                    _dup_cluster_of[_labels[_i]] = f"{_fam}#{_ci}"
            for _i in range(len(_labels)):
                _sys_cluster_of[_labels[_i]] = f"{_fam}#S{_sysid[_i] + 1}"
            _dupinfo = [(_comp, rep_of(_recs, _comp)) for _comp in _dup]
            _sg_subtypes = collections.Counter(_sysid[_rep] for _, _rep in _dupinfo)
            for _comp, _rep in _dupinfo:
                _sg = _sysid[_rep]
                _rows.append({
                    "family": _fam, "role": _recs[_rep][3],
                    "representative": _recs[_rep][0],
                    "n_strain_dupes": len(_comp),
                    "system_group": f"{_fam}#{_sg + 1}",
                    "subtypes_in_system": _sg_subtypes[_sg],
                    "is_divergent": _sg_subtypes[_sg] == 1,
                    "members": ", ".join(_recs[_i][0] for _i in _comp),
                })
        _schema_cols = ["family", "role", "representative", "n_strain_dupes",
                        "system_group", "subtypes_in_system", "is_divergent", "members"]
        _df = (pl.DataFrame(_rows).sort(["family", "representative"])
               if _rows else pl.DataFrame({c: [] for c in _schema_cols}))
        return _df, _dup_cluster_of, _sys_cluster_of

    unique_df, dup_cluster_of, sys_cluster_of = _build_unique(
        dup_id_slider.value, sys_id_slider.value, cov_slider.value)
    unique_rep_set = set(unique_df["representative"].to_list()) if unique_df.height else set()

    sum_unique = mo.md(
        f"**{unique_df.height} unique recombinases** "
        f"(strain dupes collapsed at ≥{dup_id_slider.value}% id / ≥{cov_slider.value} cov) · "
        f"**{unique_df.filter(pl.col('is_divergent')).height if unique_df.height else 0} divergent**.")
    mo.ui.table(unique_df.select(
        ['family', 'role', 'representative', 'n_strain_dupes', 'system_group',
         'subtypes_in_system', 'is_divergent']),
        selection=None, pagination=True, page_size=15)
    return (
        dup_cluster_of,
        rep_of,
        single_linkage,
        sum_unique,
        unique_df,
        unique_rep_set,
    )


@app.cell
def _(mo, unique_msa):
    _fams = sorted(unique_msa.keys()) or []
    cmp_family = mo.ui.dropdown(options=_fams, value=(_fams[0] if _fams else None), label="Family")
    cmp_metric = mo.ui.radio(
        options=["BLOSUM62 % similarity", "MAFFT % identity", "MAFFT coverage"],
        value="BLOSUM62 % similarity", label="Metric", inline=True)
    cmp_order = mo.ui.radio(
        options=["Clustered (duplicate groups)", "Plain (catalog order)"],
        value="Clustered (duplicate groups)", label="Ordering", inline=True)
    mo.vstack([mo.hstack([cmp_family, cmp_metric], justify="start", gap=2), cmp_order])
    return cmp_family, cmp_metric, cmp_order


@app.cell
def _(cmp_family, cmp_metric, mo, np, unique_msa):
    if not cmp_family.value or cmp_family.value not in unique_msa:
        cmp_labels, cmp_matrix, cmp_unit, cmp_meta = [], np.zeros((0, 0)), "", []
    else:
        _labels, _ident, _sim, _cov, cmp_meta = unique_msa[cmp_family.value]
        cmp_labels = list(_labels)
        if cmp_metric.value == "BLOSUM62 % similarity":
            cmp_matrix, cmp_unit = _sim, "% similarity"
        elif cmp_metric.value == "MAFFT coverage":
            cmp_matrix, cmp_unit = _cov, "coverage"
        else:
            cmp_matrix, cmp_unit = _ident, "% identity"
    mo.md(f"**{cmp_metric.value}** over **{len(cmp_labels)} proteins** in `{cmp_family.value}`"
          + ("" if cmp_labels else " — no sequences available."))
    return cmp_labels, cmp_matrix, cmp_unit


@app.cell
def _(
    cmp_family,
    cmp_labels,
    cmp_matrix,
    cmp_metric,
    cmp_order,
    cmp_unit,
    collections,
    dup_cluster_of,
    matrix_heatmap_plotly,
    mo,
    np,
):
    def _cluster_order(labels):
        groups = collections.defaultdict(list)
        for l in labels:
            groups[dup_cluster_of.get(l, "~singleton")].append(l)
        def _key(g):
            if g.startswith("~"):
                return (1, 0, g)
            tail = g.rsplit("#", 1)[-1]
            return (0, int(tail) if tail.isdigit() else 0, g)
        return [l for g in sorted(groups, key=_key) for l in sorted(groups[g])]

    if not cmp_labels or cmp_matrix.size == 0:
        log_heat_blosum = mo.md("*No proteins to compare.*")
        _out = log_heat_blosum
    else:
        _clustered = cmp_order.value.startswith("Clustered")
        _ord = _cluster_order(cmp_labels) if _clustered else list(cmp_labels)
        _idx = {l: cmp_labels.index(l) for l in _ord}
        _ix = [_idx[l] for l in _ord]
        _M = cmp_matrix[np.ix_(_ix, _ix)].astype(float)
        _n = len(_ord)
        _labels_short = [l.split("|")[-1] for l in _ord]
        _vmin = float(np.nanmin(_M)); _vmax = float(np.nanmax(_M))
        _is_frac = cmp_metric.value == "MAFFT coverage"
        _f = (lambda v: f"{v:.2f}") if _is_frac else (lambda v: f"{v:.1f}")
        if _clustered:
            _side = [dup_cluster_of.get(l, "singleton") for l in _ord]
            _bg = collections.defaultdict(list)
            for _i, _l in enumerate(_ord):
                _bg[dup_cluster_of.get(_l, "~singleton")].append(_i)
            _blocks = [(min(_ms), max(_ms), "#111111", 1.2) for _ms in _bg.values() if len(_ms) > 1]
        else:
            _side, _blocks = None, None
        _out = matrix_heatmap_plotly(
            _M, _labels_short, _vmin, _vmax,
            f"{cmp_unit}<br>({_f(_vmin)}–{_f(_vmax)})",
            f"{cmp_metric.value} — {cmp_family.value}" + (" · clustered" if _clustered else " · plain"),
            side_groups=_side, blocks=_blocks, height=620)
        log_heat_blosum = mo.md(
            f"**{cmp_metric.value}** over **{_n} proteins** in `{cmp_family.value}` · "
            f"range {_f(_vmin)}–{_f(_vmax)} · labels are protein accessions · "
            + ("clustered by duplicate group" if _clustered else "plain catalog order"))
    _out
    return (log_heat_blosum,)


@app.cell
def _(
    PairwiseAligner,
    cmp_family,
    cmp_labels,
    cmp_matrix,
    mo,
    np,
    recomb_seqs,
    substitution_matrices,
):
    def _pair_alignment(la, lb):
        sa = recomb_seqs.get(la.split("|")[1])
        sb = recomb_seqs.get(lb.split("|")[1])
        if not sa or not sb:
            return None
        al = PairwiseAligner()
        al.substitution_matrix = substitution_matrices.load("BLOSUM62")
        al.mode = "global"; al.open_gap_score = -11; al.extend_gap_score = -1
        aln = al.align(sa, sb)[0]
        c = aln.counts()
        tot = c.identities + c.mismatches
        pid = 100.0 * c.identities / tot if tot else 0.0
        return aln, c, pid, len(sa), len(sb)

    if not cmp_labels or cmp_matrix.size == 0:
        cmp_detail_out = mo.md("*No proteins to compare.*")
    else:
        _M = cmp_matrix.astype(float).copy()
        np.fill_diagonal(_M, np.nan)
        if not np.isfinite(_M).any():
            cmp_detail_out = mo.md("*Only one protein in this family — no pair to align.*")
        else:
            _pi, _pj = np.unravel_index(np.nanargmax(_M), _M.shape)
            _la, _lb = cmp_labels[_pi], cmp_labels[_pj]
            _res = _pair_alignment(_la, _lb)
            if _res is None:
                cmp_detail_out = mo.md("*Sequence not available for this pair.*")
            else:
                _aln, _c, _pid, _l1, _l2 = _res
                cmp_detail_out = mo.vstack([
                    mo.md(f"**Most similar pair in `{cmp_family.value}`** — "
                          f"**{_la}** ({_l1} aa) vs **{_lb}** ({_l2} aa)  \n"
                          f"BLOSUM62 score **{_aln.score:.0f}** · identity **{_pid:.1f}%** "
                          f"({_c.identities}/{_c.identities + _c.mismatches}) · gaps {_c.gaps}"),
                    mo.md(f"```\n{str(_aln)}\n```"),
                ])
    log_align = cmp_detail_out
    return (log_align,)


@app.cell
def _(mo, unique_msa):
    _famopts = sorted(unique_msa.keys(), key=lambda f: -len(unique_msa[f][0])) or []
    net_family = mo.ui.dropdown(options=_famopts, value=(_famopts[0] if _famopts else None), label="Family map")
    net_family
    return (net_family,)


@app.cell
def _(
    alt,
    cov_slider,
    dup_id_slider,
    mo,
    net_family,
    np,
    pdf,
    pl,
    single_linkage,
    unique_msa,
):
    def pcoa_2d(D):
        n = D.shape[0]
        if n <= 1:
            return np.zeros((max(n, 1), 2))
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (D ** 2) @ J
        w, v = np.linalg.eigh(B)
        idx = np.argsort(w)[::-1][:2]
        return v[:, idx] * np.sqrt(np.clip(w[idx], 0, None))

    def convex_hull(points):
        pts = sorted(set(map(tuple, points)))
        if len(pts) < 3:
            return pts
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]

    def _fam_pcoa(fam, thr_dup, thr_cov):
        labels, ident, sim, cov, recs = unique_msa[fam]
        n = len(labels)
        if n == 0:
            return None, None
        D = np.clip(100.0 - ident, 0, None)
        np.fill_diagonal(D, 0.0)
        XY = pcoa_2d(D)
        comps = single_linkage(labels, ident, cov, thr_dup, thr_cov)
        cluster_of = {i: ci for ci, comp in enumerate(comps, 1) for i in comp}
        nrows = []
        for i, lab in enumerate(labels):
            g, p = lab.split("|")[0], lab.split("|")[1]
            nrows.append({"protein": p, "genome": g,
                          "x": float(XY[i, 0]), "y": float(XY[i, 1]),
                          "cluster": f"cluster {cluster_of[i]}"})
        hrows = []
        for ci, comp in enumerate(comps, 1):
            pts = [(float(XY[i, 0]), float(XY[i, 1])) for i in comp]
            if len(pts) >= 3:
                for p in convex_hull(pts):
                    hrows.append({"x": p[0], "y": p[1], "cluster": f"cluster {ci}"})
        return pl.DataFrame(nrows), (pl.DataFrame(hrows) if hrows else None)

    if net_family.value is None:
        _out = mo.md("*No family selected.*")
    else:
        _nodes, _hull = _fam_pcoa(net_family.value, dup_id_slider.value, cov_slider.value)
        _L = []
        if _hull is not None and _hull.height:
            _L.append(alt.Chart(pdf(_hull)).mark_line(opacity=0.25, strokeWidth=1)
                      .encode(x="x:Q", y="y:Q", color=alt.Color("cluster:N", legend=None),
                              detail="cluster:N"))
        _L.append(alt.Chart(pdf(_nodes)).mark_circle(size=110, filled=True, opacity=0.9)
                  .encode(x=alt.X("x:Q", title="PCoA 1"), y=alt.Y("y:Q", title="PCoA 2"),
                          color=alt.Color("cluster:N",
                                          legend=alt.Legend(title="duplicate cluster",
                                                            labelFontSize=8, symbolLimit=100)),
                          tooltip=["protein", "genome", "cluster"]))
        _out = mo.ui.altair_chart(
            chart_selection=False, legend_selection=False,
            chart=alt.layer(*_L).properties(
                width=640, height=470,
                title=f"{net_family.value} — PCoA of {len(_nodes)} proteins "
                      f"(hulls = duplicates ≥{dup_id_slider.value}% id / ≥{cov_slider.value} cov)").interactive())
    _out
    return


@app.cell
def _(alt, mo, pdf, result, unique_df):
    if unique_df.height:
        _uf = (unique_df.group_by(["family", "is_divergent"]).len()
               .rename({"len": "n"}).sort("n", descending=True))
        _chart = (alt.Chart(pdf(_uf))
                  .mark_bar()
                  .encode(
                      x=alt.X("n:Q", title="unique recombinases"),
                      y=alt.Y("family:N", sort="-x", title=None),
                      color=alt.Color("is_divergent:N", title="divergent subtype",
                                      scale=alt.Scale(domain=[True, False],
                                                      range=["#e45756", "#4c78a8"])),
                      tooltip=["family", "is_divergent", "n"])
                  .properties(height=max(140, 34 * _uf["family"].n_unique()), width=460,
                              title="Unique recombinases per family (after deduplication)"))
        _out = mo.ui.altair_chart(_chart)
        _tot = unique_df.group_by("family").len().sort("len", descending=True)
        sum_unique_fam = mo.md(
            f"**{unique_df.height} unique recombinases** across **{_tot.height} families** — "
            + " · ".join(f"{r['family']} {r['len']}" for r in _tot.iter_rows(named=True))
            + ". Families named `SSAP(PSI:…)` are PSI-BLAST-only (no Pfam profile) and are "
              "deduplicated separately.")
    else:
        sum_unique_fam = mo.md("*No unique recombinases at the current thresholds.*")
        _out = sum_unique_fam
    result(_out)
    return (sum_unique_fam,)


@app.cell(hide_code=True)
def _step8_md(mo):
    mo.md(r"""
    ## Step 8 — Identify putative SSAP↔exonuclease/Gam operons

    Each catalog protein is mapped to its pyrodigal CDS coordinates
    (`db/genes.tsv`) and every genome is scanned for an SSAP within an adjustable
    **gap** (default 5 kb) of an exonuclease (YqaJ) or host-nuclease inhibitor (Gam) on
    the same contig, recording the gap and orientation (`SAME` = operon-like).
    """)
    return


@app.cell
def _recomb_loci(GENES_TSV, mo, pl, recomb_df):

    def _load_loci():
        _genes = pl.read_csv(GENES_TSV, separator="\t", infer_schema_length=0) if GENES_TSV.exists() else None
        if _genes is not None and _genes.height:
            _genes = _genes.with_columns(pl.col("start").cast(pl.Int64, strict=False),
                                         pl.col("end").cast(pl.Int64, strict=False))
        _cat = recomb_df.select(["genome", "protein", "family", "role"])
        if _genes is None or not _genes.height:
            return pl.DataFrame(schema={"genome": pl.Utf8, "protein": pl.Utf8, "contig": pl.Utf8,
                                        "start": pl.Int64, "end": pl.Int64, "strand": pl.Utf8,
                                        "family": pl.Utf8, "role": pl.Utf8})
        return _genes.join(_cat, on=["genome", "protein"], how="inner")

    recomb_loci_df = _load_loci()
    log_loci = mo.md(f"Placed **{recomb_loci_df.height}** recombineering proteins on CDS coordinates.")
    return log_loci, recomb_loci_df


@app.cell
def _coloc_controls(mo):
    gap_slider = mo.ui.slider(0, 20000, step=250, value=5000,
                              label="Max SSAP↔partner gap (bp)", show_value=True)
    unique_only_toggle = mo.ui.switch(value=False, label="Unique operons only (drop strain dupes)")
    mo.hstack([gap_slider, unique_only_toggle], justify="start", gap=2)
    return gap_slider, unique_only_toggle


@app.cell
def _coloc_compute(
    collections,
    gap_slider,
    mo,
    pl,
    recomb_loci_df,
    unique_only_toggle,
    unique_rep_set,
):

    def _find_operons(_gap_max):
        _by_genome = collections.defaultdict(list)
        for _r in recomb_loci_df.iter_rows(named=True):
            _by_genome[_r["genome"]].append(_r)
        _rows = []
        for _genome, _genes in _by_genome.items():
            _ssaps = [g for g in _genes if g["role"] == "SSAP"]
            _partners = [g for g in _genes if g["role"] in ("EXO", "accessory")]
            for _s in _ssaps:
                for _e in _partners:
                    if _s["contig"] != _e["contig"]:
                        continue
                    _lo, _hi = (_s, _e) if _s["start"] <= _e["start"] else (_e, _s)
                    _gap = _hi["start"] - _lo["end"]
                    if _gap > _gap_max:
                        continue
                    _rows.append({
                        "genome": _genome, "contig": _s["contig"],
                        "ssap_family": _s["family"], "ssap_protein": _s["protein"],
                        "partner_role": _e["role"], "partner_family": _e["family"],
                        "partner_protein": _e["protein"],
                        "gap": _gap, "overlap": _gap < 0,
                        "orientation": "SAME" if _s["strand"] == _e["strand"] else "DIVERGENT",
                        "ssap_start": _s["start"], "ssap_end": _s["end"],
                        "partner_start": _e["start"], "partner_end": _e["end"],
                        "ssap_unique": f"{_genome}|{_s['protein']}" in unique_rep_set,
                    })
        _oschema = ["genome", "contig", "ssap_family", "ssap_protein", "partner_role",
                    "partner_family", "partner_protein", "gap", "overlap", "orientation",
                    "ssap_start", "ssap_end", "partner_start", "partner_end", "ssap_unique"]
        _df = pl.DataFrame(_rows, schema=_oschema) if _rows else pl.DataFrame(schema=_oschema)
        return _df.sort(["genome", "gap"]) if _df.height else _df

    _operons_all = _find_operons(gap_slider.value)
    operons_df = (_operons_all.filter(pl.col("ssap_unique"))
                  if (unique_only_toggle.value and _operons_all.height) else _operons_all)
    sum_operons = mo.md(
        f"**{operons_df.height} SSAP↔partner pair(s)** within {gap_slider.value:,} bp across "
        f"{operons_df['genome'].n_unique() if operons_df.height else 0} phages"
        + (" (unique only)" if unique_only_toggle.value else "")
        + f" · {operons_df.filter(pl.col('orientation') == 'SAME').height if operons_df.height else 0} SAME-strand"
        + f" · {operons_df.filter(pl.col('overlap')).height if operons_df.height else 0} overlapping")
    mo.ui.table(operons_df.select(['genome', 'ssap_family', 'ssap_protein', 'partner_role',
                                   'partner_family', 'partner_protein', 'gap', 'orientation']),
                selection=None, pagination=True, page_size=15)
    return operons_df, sum_operons


@app.cell
def _(
    ACCESSORY_FAMILIES,
    EXO_FAMILIES,
    GENES_TSV,
    GraphicFeature,
    GraphicRecord,
    SSAP_FAMILIES,
    collections,
    hmm_hits_all,
    mo,
    pl,
    plt,
    recomb_loci_df,
    recomb_seqs,
):
    ROLE_COLORS = {"SSAP": "#4c78a8", "EXO": "#e45756", "accessory": "#54a24b", "other": "#f58518"}

    recomb_locus_map = {(r["genome"], r["protein"]): (r["family"], r["role"])
                        for r in recomb_loci_df.iter_rows(named=True)}

    all_genes_df = (pl.read_csv(GENES_TSV, separator="\t", infer_schema_length=0)
                    if GENES_TSV.exists() else pl.DataFrame())
    if all_genes_df.height:
        all_genes_df = all_genes_df.with_columns(pl.col("start").cast(pl.Int64, strict=False),
                                                 pl.col("end").cast(pl.Int64, strict=False))

    _genes_by_genome = collections.defaultdict(list)
    if all_genes_df.height:
        for _r in all_genes_df.iter_rows(named=True):
            _genes_by_genome[_r["genome"]].append(_r)

    def viewer_cds_in_region(genome, contig, lo, hi):
        _out = []
        for _g in _genes_by_genome.get(genome, []):
            if _g["contig"] != contig:
                continue
            if _g["end"] < lo or _g["start"] > hi:
                continue
            _out.append({"start": _g["start"], "end": _g["end"],
                         "strand": _g["strand"], "protein": _g["protein"], "product": ""})
        return _out

    def viewer_render_region(genome, contig, lo, hi, title):
        lo = max(1, lo)
        _gfeats = []
        for _f in viewer_cds_in_region(genome, contig, lo, hi):
            _key = (genome, _f["protein"])
            if _key in recomb_locus_map:
                _fam, _role = recomb_locus_map[_key]
                _color, _label = ROLE_COLORS.get(_role, "#f58518"), f"{_fam} · {_f['protein']}"
            else:
                _color, _label = "#c9c9c9", None
            _gfeats.append(GraphicFeature(
                start=_f["start"], end=_f["end"],
                strand=(1 if _f["strand"] == "+" else -1), color=_color, label=_label))
        _rec = GraphicRecord(sequence_length=(hi - lo + 1), first_index=lo, features=_gfeats)
        _fig, _ax = plt.subplots(figsize=(13, 3.2))
        _rec.plot(ax=_ax, with_ruler=True)
        _ax.set_title(title, fontsize=10)
        return _fig

    def viewer_domain_fig(genome, protein):
        _doms = hmm_hits_all.filter((pl.col("genome") == genome) & (pl.col("protein") == protein))
        if not _doms.height:
            return None
        _tlen = int(_doms["tlen"][0])
        _feats = []
        for _d in _doms.iter_rows(named=True):
            _col = ("#4c78a8" if _d["family"] in SSAP_FAMILIES else
                    "#e45756" if _d["family"] in EXO_FAMILIES else
                    "#54a24b" if _d["family"] in ACCESSORY_FAMILIES else "#f58518")
            _feats.append(GraphicFeature(start=_d["ali_from"], end=_d["ali_to"], strand=0,
                                         color=_col,
                                         label=f"{_d['family']} {_d['ali_from']}-{_d['ali_to']} (E={_d['i_evalue']:.0e})"))
        _rec = GraphicRecord(sequence_length=_tlen, features=_feats)
        _fig, _ax = plt.subplots(figsize=(11, 1.5))
        _rec.plot(ax=_ax, with_ruler=True)
        _ax.set_title(f"{protein} — {_tlen} aa", fontsize=9)
        return _fig

    def viewer_seq_block(protein):
        _s = recomb_seqs.get(protein)
        if not _s:
            return mo.md(f"*sequence for {protein} not loaded*")
        _wrapped = "\n".join(_s[i:i + 60] for i in range(0, len(_s), 60))
        return mo.md(f"**{protein}** ({len(_s)} aa)\n```\n{_wrapped}\n```")

    return viewer_domain_fig, viewer_render_region, viewer_seq_block


@app.cell
def _(mo, operons_df):
    def _operon_options(df):
        _opts = {}
        for _i, _r in enumerate(df.iter_rows(named=True)):
            _lab = (f"{_r['genome']} · {_r['ssap_family']} {_r['ssap_protein']}"
                    f" + {_r['partner_family']} {_r['partner_protein']} ({_r['partner_role']})"
                    f"  (gap {_r['gap']}bp, {_r['orientation']})")
            _opts[_lab] = _i
        return _opts

    _opts = _operon_options(operons_df) if operons_df.height else {}
    operon_pick = mo.ui.dropdown(options=_opts, value=(next(iter(_opts)) if _opts else None),
                                 label="Operon to inspect", searchable=True)
    flank_slider = mo.ui.slider(1000, 25000, step=1000, value=8000,
                                label="Flank window (bp)", show_value=True)
    mo.hstack([operon_pick, flank_slider], justify="start", gap=2)
    return flank_slider, operon_pick


@app.cell
def _(flank_slider, mo, operon_pick, operons_df, result, viewer_render_region):
    def _render_operon(op, flank):
        lo = max(1, min(op["ssap_start"], op["partner_start"]) - flank)
        hi = max(op["ssap_end"], op["partner_end"]) + flank
        return viewer_render_region(
            op["genome"], op["contig"], lo, hi,
            f"{op['genome']}  ({op['ssap_family']}+{op['partner_family']}, {op['orientation']})")

    if operon_pick.value is None:
        view_map_out = mo.md("*No operons to display at the current gap threshold.*")
    else:
        _op = operons_df.row(operon_pick.value, named=True)
        _link = f"https://www.ncbi.nlm.nih.gov/nuccore/{_op['genome']}/"
        view_map_out = mo.vstack([
            _render_operon(_op, flank_slider.value),
            mo.md(f"[↗ NCBI record for {_op['genome']}]({_link})  ·  contig `{_op['contig']}`  ·  "
                  f"*phage CDS predicted with pyrodigal (no regulatory annotations)*"),
        ])
    result(view_map_out)
    return


@app.cell
def _(
    mo,
    operon_pick,
    operons_df,
    result,
    viewer_domain_fig,
    viewer_seq_block,
):
    if operon_pick.value is None:
        view_domain_out = mo.md("")
    else:
        _op = operons_df.row(operon_pick.value, named=True)
        _panels = []
        for _which, _prot in [("SSAP", _op["ssap_protein"]), ("Partner", _op["partner_protein"])]:
            _f = viewer_domain_fig(_op["genome"], _prot)
            if _f is not None:
                _panels.append(mo.md(f"**{_which} domain map**"))
                _panels.append(_f)
        view_domain_out = mo.vstack(_panels + [mo.accordion({
            f"SSAP sequence ({_op['ssap_protein']})": viewer_seq_block(_op["ssap_protein"]),
            f"Partner sequence ({_op['partner_protein']})": viewer_seq_block(_op["partner_protein"]),
        })])
    result(view_domain_out)
    return


@app.cell
def _(mo, recomb_loci_df):
    _gopts = sorted(recomb_loci_df["genome"].unique().to_list())
    browse_genome = mo.ui.dropdown(
        options={a: a for a in _gopts},
        value=(_gopts[0] if _gopts else None),
        label="Phage", searchable=True)
    browse_flank = mo.ui.slider(1000, 25000, step=1000, value=8000,
                                label="Flank (bp)", show_value=True)
    mo.hstack([browse_genome, browse_flank], justify="start", gap=2)
    return browse_flank, browse_genome


@app.cell
def _(browse_genome, mo, pl, recomb_loci_df):
    _prots = (recomb_loci_df.filter(pl.col("genome") == browse_genome.value)
              if browse_genome.value else recomb_loci_df.head(0))
    _popts = {f"{r['family']} · {r['protein']} ({r['role']})": r["protein"]
              for r in _prots.iter_rows(named=True)}
    browse_protein = mo.ui.dropdown(options=_popts,
                                    value=(next(iter(_popts)) if _popts else None),
                                    label="Recombinase")
    browse_protein
    return (browse_protein,)


@app.cell
def _(
    browse_flank,
    browse_genome,
    browse_protein,
    mo,
    pl,
    recomb_loci_df,
    result,
    viewer_domain_fig,
    viewer_render_region,
    viewer_seq_block,
):
    if browse_genome.value is None or browse_protein.value is None:
        browse_out = mo.md("*No recombinase selected.*")
    else:
        _row = recomb_loci_df.filter(
            (pl.col("genome") == browse_genome.value)
            & (pl.col("protein") == browse_protein.value)).row(0, named=True)
        _lo = max(1, _row["start"] - browse_flank.value)
        _hi = _row["end"] + browse_flank.value
        _fig = viewer_render_region(
            _row["genome"], _row["contig"], _lo, _hi,
            f"{_row['genome']}  {_lo:,}-{_hi:,}  [{_row['family']} {_row['protein']}]")
        _dfig = viewer_domain_fig(_row["genome"], _row["protein"])
        _link = f"https://www.ncbi.nlm.nih.gov/nuccore/{_row['genome']}/"
        _parts = [_fig]
        if _dfig is not None:
            _parts += [mo.md("**Domain map**"), _dfig]
        _parts += [viewer_seq_block(_row["protein"]),
                   mo.md(f"[↗ NCBI record]({_link})")]
        browse_out = mo.vstack(_parts)
    result(browse_out)
    return


@app.cell(hide_code=True)
def _step9_md(mo):
    mo.md(r"""
    ## Step 9 — Rank phages by genome quality + prophage-region support

    Following deduplication, each representative phage is scored by a weighted sum of
    transparent, normalised components:

    1. **genome quality** (single-record completeness + length)
    2. **prophage-region support** (how many prophage regions point at this phage —
       BLASTn top-hit or PHASTEST suggestion)
    3. **has an exonuclease partner**
    4. **has a Gam partner**
    5. **number of unique recombinases**

    The result is a ranked recombinase shortlist and a ranked system (operon) shortlist.
    """)
    return


@app.cell
def _rank_controls(mo):
    _w_defaults = {"quality": 0.25, "regions": 0.25, "exo": 0.25, "gam": 0.05, "nrec": 0.20}
    w_quality = mo.ui.slider(0.0, 1.0, step=0.05, value=_w_defaults["quality"],
                             label="importance · genome quality")
    w_regions = mo.ui.slider(0.0, 1.0, step=0.05, value=_w_defaults["regions"],
                             label="importance · prophage regions → phage")
    w_exo = mo.ui.slider(0.0, 1.0, step=0.05, value=_w_defaults["exo"],
                         label="importance · has EXO partner")
    w_gam = mo.ui.slider(0.0, 1.0, step=0.05, value=_w_defaults["gam"],
                         label="importance · has Gam")
    w_nrec = mo.ui.slider(0.0, 1.0, step=0.05, value=_w_defaults["nrec"],
                          label="importance · # unique recombinases")
    mo.vstack([
        mo.hstack([w_quality, w_regions, w_exo, w_gam, w_nrec], justify="start", gap=2, wrap=True),
        mo.md("Sliders are **relative importance**: normalised downstream so weights sum to 1.000."),
    ])
    return w_exo, w_gam, w_nrec, w_quality, w_regions


@app.cell
def _ranking(
    REGIONS_TSV,
    REGION_VIRAL_BLAST_TSV,
    mo,
    np,
    operons_df,
    parse_blastn_tsv,
    phage_meta_df,
    pl,
    re,
    rep_of,
    single_linkage,
    unique_df,
    unique_msa,
    w_exo,
    w_gam,
    w_nrec,
    w_quality,
    w_regions,
):

    def _pointers():
        from collections import Counter
        _cnt = Counter()
        _cnt_bl = Counter()
        _cnt_ph = Counter()
        if REGION_VIRAL_BLAST_TSV.exists():
            for _row in parse_blastn_tsv(REGION_VIRAL_BLAST_TSV):
                _m = re.search(r"(NC_\d+)", _row["sseqid"])
                if _m:
                    _cnt[_m.group(1)] += 1
                    _cnt_bl[_m.group(1)] += 1
        if REGIONS_TSV.exists():
            _rdf = pl.read_csv(REGIONS_TSV, separator="\t", infer_schema_length=0)
            for _r in _rdf.filter(pl.col("most_common_phage") != "").iter_rows(named=True):
                _m = re.search(r"(NC_\d+)", _r["most_common_phage"])
                if _m:
                    _cnt[_m.group(1)] += 1
                    _cnt_ph[_m.group(1)] += 1
        return _cnt, _cnt_bl, _cnt_ph

    region_count, count_blast, count_phastest = _pointers()

    def _minmax(_vals):
        _v = [x for x in _vals if x is not None and np.isfinite(x)]
        if not _v:
            return lambda x: 0.0
        _lo, _hi = min(_v), max(_v)
        if _hi <= _lo:
            return lambda x: 1.0
        return lambda x: 0.0 if x is None or not np.isfinite(x) else (float(x) - _lo) / (_hi - _lo)

    _wraw = {"quality": w_quality.value, "regions": w_regions.value, "exo": w_exo.value,
             "gam": w_gam.value, "nrec": w_nrec.value}
    _wtot = sum(_wraw.values())
    w = ({k: 1.0 / len(_wraw) for k in _wraw} if _wtot <= 0
         else {k: v / _wtot for k, v in _wraw.items()})

    _w_labels = [("quality", "genome quality"), ("regions", "prophage regions → phage"),
                 ("exo", "has EXO partner"), ("gam", "has Gam"),
                 ("nrec", "# unique recombinases")]
    _weight_md = mo.md(
        "| criterion | slider | normalised weight |\n|---|---|---|\n"
        + "\n".join(f"| {lbl} | {_wraw[k]:.2f} | **{w[k]:.1%}** |" for k, lbl in _w_labels)
        + f"\n| **total** | **{_wtot:.2f}** | **100%** |")

    _meta_rows = {_r["accession"]: _r for _r in phage_meta_df.iter_rows(named=True)}
    _reps = sorted(phage_meta_df["accession"].to_list())

    _len_n = _minmax([_meta_rows.get(a, {}).get("length") or 0 for a in _reps])
    _quality = {}
    for _a in _reps:
        _r = _meta_rows.get(_a, {})
        _length = _r.get("length") or 0
        _single = (_r.get("n_records") or 0) == 1
        _quality[_a] = 0.6 * (1.0 if _single else 0.5) + 0.4 * _len_n(_length)
    _q_n = _minmax(list(_quality.values()))
    _r_n = _minmax([region_count.get(a, 0) for a in _reps])

    _n_unique = {}
    for _r in unique_df.iter_rows(named=True):
        _g = _r["representative"].split("|")[0]
        _n_unique[_g] = _n_unique.get(_g, 0) + 1
    _n_n = _minmax([_n_unique.get(a, 0) for a in _reps])

    _exo = set(operons_df.filter(pl.col("partner_role") == "EXO")["genome"].to_list()) if operons_df.height else set()
    _gam = set(operons_df.filter(pl.col("partner_role") == "accessory")["genome"].to_list()) if operons_df.height else set()

    _rows = []
    for _a in _reps:
        _r = _meta_rows.get(_a, {})
        _rows.append({
            "genome": _a, "name": _r.get("name", ""),
            "n_records": _r.get("n_records"), "length": _r.get("length"),
            "quality": round(_q_n(_quality[_a]), 3),
            "n_regions": region_count.get(_a, 0),
            "n_regions_blastn": count_blast.get(_a, 0),
            "n_regions_phastest": count_phastest.get(_a, 0),
            "regions_norm": round(_r_n(region_count.get(_a, 0)), 3),
            "has_exo": _a in _exo, "has_gam": _a in _gam,
            "n_unique": _n_unique.get(_a, 0), "n_unique_norm": round(_n_n(_n_unique.get(_a, 0)), 3),
            "score": round(w["quality"] * _q_n(_quality[_a])
                           + w["regions"] * _r_n(region_count.get(_a, 0))
                           + w["exo"] * float(_a in _exo)
                           + w["gam"] * float(_a in _gam)
                           + w["nrec"] * _n_n(_n_unique.get(_a, 0)), 3),
        })
    genome_scores = pl.DataFrame(_rows).sort("score", descending=True)

    recombinase_shortlist = (unique_df
        .with_columns(pl.col("representative").str.split("|").list.first().alias("genome"))
        .join(genome_scores.select(["genome", "score", "n_regions", "has_exo", "has_gam"]),
              on="genome", how="left")
        .sort("score", descending=True))

    _dup_rep = {}
    for _fam, (_lab, _id, _sim, _cov, _recs) in unique_msa.items():
        for _comp in single_linkage(_lab, _id, _cov, 90, 0.8):
            _rep = rep_of(_recs, _comp)
            for _i in _comp:
                _dup_rep[_lab[_i]] = _lab[_rep]

    if operons_df.height:
        _score_of = {_r["genome"]: _r["score"] for _r in genome_scores.iter_rows(named=True)}
        _best = {}
        for _r in operons_df.iter_rows(named=True):
            _lab = f"{_r['genome']}|{_r['ssap_protein']}"
            _key = (_dup_rep.get(_lab, _lab), _r["partner_family"], _r["orientation"])
            _rec = dict(_r, ssap_dup_rep=_dup_rep.get(_lab, _lab),
                        genome_score=_score_of.get(_r["genome"], 0.0))
            if _key not in _best or _rec["genome_score"] > _best[_key]["genome_score"]:
                _best[_key] = _rec
        system_shortlist = pl.DataFrame(list(_best.values())).sort("genome_score", descending=True)
    else:
        system_shortlist = pl.DataFrame(schema={"genome": pl.Utf8, "ssap_protein": pl.Utf8,
                                                "partner_family": pl.Utf8, "genome_score": pl.Float64})

    sum_rank = mo.md(
        f"Score = Σ (normalised component × weight) · **{recombinase_shortlist.height} recombinases** and "
        f"**{system_shortlist.height} deduplicated systems** ranked.")
    mo.vstack([
        mo.md("#### Ranked phages & shortlists"),
        _weight_md,
        mo.ui.tabs({
            "Recombinases (unique SSAP reps)": mo.ui.table(
                recombinase_shortlist.select(
                    ["score", "family", "representative", "n_strain_dupes", "system_group",
                     "is_divergent", "genome", "n_regions", "has_exo", "has_gam"]),
                selection=None, pagination=True, page_size=12),
            "Systems (deduplicated operons)": mo.ui.table(
                system_shortlist.select(
                    ["genome_score", "genome", "ssap_dup_rep", "ssap_family", "partner_role",
                     "partner_family", "gap", "orientation"]),
                selection=None, pagination=True, page_size=12),
        }),
        mo.ui.table(genome_scores, selection=None, pagination=True, page_size=12),
    ])
    return genome_scores, recombinase_shortlist, sum_rank, system_shortlist


@app.cell
def _results_summary(
    log_align,
    log_dedup,
    log_genomad,
    log_heat_blosum,
    log_hosts,
    log_identity,
    log_loci,
    log_msa,
    log_phage_set,
    log_phastest,
    log_proteins,
    log_regions,
    mo,
    result,
    sum_catalog,
    sum_compare,
    sum_dedup,
    sum_hmm_exo,
    sum_hmm_rec,
    sum_merged_exo,
    sum_merged_rec,
    sum_operons,
    sum_psi_exo,
    sum_psi_rec,
    sum_rank,
    sum_superfam,
    sum_superfam_exo,
    sum_unique,
    sum_unique_fam,
):
    result(mo.accordion({"Results": mo.vstack([
        log_genomad, log_phastest, log_regions, log_dedup, sum_compare,
        log_phage_set, log_hosts, log_identity, sum_dedup, log_proteins,
        sum_hmm_rec, sum_psi_rec, sum_merged_rec, sum_superfam,
        sum_hmm_exo, sum_psi_exo, sum_merged_exo, sum_superfam_exo, sum_catalog,
        log_msa, sum_unique, sum_unique_fam, log_heat_blosum, log_align, log_loci, sum_operons, sum_rank,
    ])}), title="Phage workflow results")
    return


@app.cell
def _exports(
    genome_scores,
    mo,
    operons_df,
    phage_meta_df,
    recomb_df,
    recomb_loci_df,
    recombinase_shortlist,
    region_blast_df,
    regions_df,
    result,
    system_shortlist,
    unique_df,
):

    def _dl(_df, _name):
        return mo.download(data=_df.write_csv().encode(), filename=_name,
                           mimetype="text/csv", label=f"⬇ {_name} ({_df.height})")

    result(mo.vstack([
        mo.md("## Exports\n\nDownload the **current, filtered** tables as CSV."),
        mo.md("**Regions / BLAST / phages**"),
        mo.hstack([_dl(regions_df, "prophage_regions.csv"),
                   _dl(region_blast_df, "region_blastn_top_hits.csv"),
                   _dl(phage_meta_df, "phage_metadata.csv")], justify="start", gap=1),
        mo.md("**Searches / catalog**"),
        mo.hstack([_dl(recomb_df, "recombineering_catalog.csv"),
                   _dl(unique_df, "unique_recombinases.csv"),
                   _dl(recomb_loci_df, "recombining_loci.csv"),
                   _dl(operons_df, "ssap_partner_pairs.csv")], justify="start", gap=1),
        mo.md("**Ranked shortlists**"),
        mo.hstack([_dl(genome_scores, "phage_scores.csv"),
                   _dl(recombinase_shortlist, "recombinase_shortlist.csv"),
                   _dl(system_shortlist, "system_shortlist.csv")], justify="start", gap=1),
    ]))
    return


if __name__ == "__main__":
    app.run()
