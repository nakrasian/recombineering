import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # _Xanthobacter autotrophicus_ Recombinase Discovery Workflow
    ## Overview
    This notebook outlines the process of analyzing _X. autotrophicus'_ genomes for putative recombinases. This is done using **PSI-BLAST** and **HMMER**, tools that use libraries of known protein sequences to find distantly related sequences in a given database. Following this protein search, the resulting hits are filtered based off of their similarity to each other in order to deduplicate and reduce the total number of recombinases to test. Exonucleases, necessary for dsDNA-based recombination, are also explored using the same protein search and refinement method. Following these sets of recombinases and exonucleases, individual genomes can be searched to identify potential operons that express both the exonuclease and recombinase.

    ### Outline:
    1. Import desired genomic data in the form of .txt or .faa
    2. Compile databases for PSI-BLAST and HMMER search
    3. Search for recombinases and exonucleases using PSI-BLAST and HMMER
    4. Deduplicate recombinases based off of their % similarities
    5. Identify putative operons containing combinations of recombinases and exonucleases
    6. Evaluate recombinases and exonucleases based off of their genomes relation to _X. autotrophicus_ GJ10

    ### Key Outputs:
    1. Create a list of unique recombinases for bench-top testing in oligonucleotide based recombination
    2. Create a list of unique recombinase-exonuclease operons for dsDNA based recombination

    ---
    """)
    return


@app.cell
def core_imports():

    import polars as pl
    import numpy as np
    import altair as alt
    import re
    import io
    import os
    import json
    import gzip
    import zipfile
    import shutil
    import collections
    import subprocess
    from pathlib import Path
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from Bio import SeqIO
    from Bio.Align import PairwiseAligner, substitution_matrices
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from dna_features_viewer import GraphicFeature, GraphicRecord


    return (
        GraphicFeature,
        GraphicRecord,
        PairwiseAligner,
        Path,
        SeqIO,
        alt,
        collections,
        gzip,
        io,
        json,
        leaves_list,
        linkage,
        matplotlib,
        np,
        pd,
        pl,
        plt,
        re,
        shutil,
        subprocess,
        substitution_matrices,
        zipfile,
    )


@app.cell
def pdf_util(matplotlib, mo, pd, plt):

    def pdf(df):
        """polars -> pandas for altair, avoiding the pyarrow path."""
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
        """Interactive Plotly heatmap on a *numeric* (linear) axis.

        Positions are 0..n-1 and the labels ride along as tick text, so repeated
        labels (e.g. one protein accession found in several genomes) never collapse
        cells -- a categorical axis deduplicates repeated category names and silently
        shrinks the matrix, which desyncs it from the block outlines.

        side_groups: per-row/col group label -> colour strips top+left.
        blocks: list of (start, end, colour, width) index ranges outlined on the matrix.
        """
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


    return matrix_heatmap_plotly, pdf, result, results_box


@app.cell
def parse_util(Path, SeqIO):
    def seed_id(qseqid):
        """Seed accession from a UniProt-style BLAST qseqid (sp|P03698|NAME)."""
        parts = qseqid.split("|")
        tok = parts[1] if len(parts) >= 2 else parts[0]
        return tok.split()[0]

    def load_protein_seqs(faa, wanted_ids):
        """Protein id -> sequence for `genome|protein` style FASTA ids."""
        seqs = {}
        if not Path(faa).exists() or not wanted_ids:
            return seqs
        for rec in SeqIO.parse(str(faa), "fasta"):
            parts = rec.id.split("|")
            pid = parts[1] if len(parts) >= 2 else rec.id
            if pid in wanted_ids and pid not in seqs:
                seqs[pid] = str(rec.seq)
        return seqs

    def parse_psiblast_tsv(path):
        """Parse BLAST/PSI-BLAST outfmt '6 std qcovhsp' into row dicts."""
        out = []
        with open(path) as fh:
            for line in fh:
                if not line.strip() or line.startswith(("Search", "#")):
                    continue
                c = line.rstrip("\n").split("\t")
                if len(c) < 12:
                    continue
                genome, _, protein = c[1].partition("|")
                try:
                    out.append({
                        "seed": seed_id(c[0]), "genome": genome,
                        "protein": protein or c[1],
                        "pident": float(c[2]), "aln_len": int(c[3]),
                        "evalue": float(c[10]), "bitscore": float(c[11]),
                        "qcovhsp": int(c[12]) if len(c) > 12 else None,
                    })
                except ValueError:
                    continue
        return out


    return load_protein_seqs, parse_psiblast_tsv, seed_id


@app.cell(hide_code=True)
def setup_md(mo):
    mo.md(rf"""
    ## **Data Processing Pipeline:**
    ### 1. Import desired genomic data in the form of .txt or .faa

    This workflow requires a curated list of genomes to access the respective genome sequences and protein databases.
    - **Genome set** — drop either an *accession list* (a `.txt`/`.tsv` whose
      first column contains `GCA_`/`GCF_` accessions) or a *genome FASTA*
      (`.fna`/`.fasta`, headers like `>GCF_XXXX.1|contig ...`). Accessions are
      extracted with the regex `GC[AF]_\d+\.\d+`.

    If nothing is dropped, the notebook falls back to the bundled
    `xanthobacter_121_accessions.txt`
    """)
    return


@app.cell
def config_ui(Path, mo):
    data_root = mo.ui.text(
        value=str(Path(__file__).resolve().parent),
        label="Data root",
        full_width=True,
    )
    dataset_slug = mo.ui.text(
        value="xanthobacter_121",
        label="Dataset name (folder under data/)",
    )
    genome_input = mo.ui.file(
        kind="area",
        label="Drop an accession list (.txt/.tsv), a genome FASTA (.fna/.fasta), or a protein FASTA (.faa)",
    )
    force_rebuild = mo.ui.switch(
        value=False,
        label="Force rebuild (overwrite this dataset's artifacts)",
    )
    mo.vstack(
        [
            data_root,
            mo.hstack([dataset_slug, force_rebuild], justify="start", gap=2),
            genome_input,
        ]
    )
    return data_root, dataset_slug, force_rebuild, genome_input


@app.cell
def root_path(Path, data_root, dataset_slug, force_rebuild, shutil):

    ROOT = Path(data_root.value).expanduser().resolve()
    DATASET_DIR = ROOT / "data" / (dataset_slug.value.strip() or "dataset")

    if force_rebuild.value and DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    IN_DIR = DATASET_DIR / "input"
    GENOMES_DIR = DATASET_DIR / "genomes"
    META_DIR = DATASET_DIR / "meta"
    DB_DIR = DATASET_DIR / "db"
    HMM_DIR = DB_DIR / "hmms"
    QUERY_DIR = DB_DIR / "queries"
    RESULTS_DIR = DATASET_DIR / "results"
    ANI_DIR = DATASET_DIR / "ani"
    MSA_DIR = DATASET_DIR / "msa"

    GENOMES_FNA = DATASET_DIR / "genomes.fna"
    PROTEIN_FAA = DB_DIR / "proteins.faa"
    BLAST_DB = DB_DIR / "proteins"
    GENOMES_TSV = META_DIR / "genomes.tsv"
    ASSEMBLY_REPORT = META_DIR / "assembly_data_report.jsonl"

    for _d in (IN_DIR, GENOMES_DIR, META_DIR, DB_DIR, HMM_DIR, QUERY_DIR,
               RESULTS_DIR, ANI_DIR, MSA_DIR):
        _d.mkdir(parents=True, exist_ok=True)
    return (
        ANI_DIR,
        ASSEMBLY_REPORT,
        BLAST_DB,
        DATASET_DIR,
        GENOMES_DIR,
        GENOMES_FNA,
        GENOMES_TSV,
        HMM_DIR,
        IN_DIR,
        MSA_DIR,
        PROTEIN_FAA,
        QUERY_DIR,
        RESULTS_DIR,
        ROOT,
    )


@app.cell
def parse_genome_set(
    DATASET_DIR,
    IN_DIR,
    ROOT,
    force_rebuild,
    genome_input,
    mo,
    re,
):

    ACC_RE = re.compile(r"GC[AF]_\d+\.\d+")

    def _read_dropped(file_val):
        if not file_val:
            return None, None
        f = file_val[0]
        raw = f.contents
        text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        return f.name, text

    def _accs_from_text(text):
        accs = []
        for line in text.splitlines():
            if not line.strip():
                continue
            if line.startswith(">"):
                m = ACC_RE.search(line)
            elif line.lower().startswith("accession"):
                continue
            else:
                m = ACC_RE.search(line.split("\t")[0])
            if m:
                accs.append(m.group(0))
        return sorted(set(accs))

    def _parse_input(file_val, root):
        """Return (accs, mode, src, name, text); mode in accessions|genomes|proteins."""
        name, text = _read_dropped(file_val)
        if name is None:
            fb = root / "xanthobacter_121_accessions.txt"
            accs = _accs_from_text(fb.read_text()) if fb.exists() else []
            return accs, "accessions", f"fallback ({fb.name})", None, None
        lower = name.lower()
        if lower.endswith((".faa", ".faa.gz")):
            return _accs_from_text(text), "proteins", f"protein FASTA ({name})", name, text
        if lower.endswith((".fna", ".fasta", ".fa", ".fas")):
            return _accs_from_text(text), "genomes", f"genome FASTA ({name})", name, text
        return _accs_from_text(text), "accessions", f"accession list ({name})", name, text

    genome_accs, input_mode, genome_src, input_name, input_text = _parse_input(genome_input.value, ROOT)
    genome_set = set(genome_accs)

    input_fasta_path = None
    if input_mode in ("genomes", "proteins") and input_text is not None and input_name:
        input_fasta_path = IN_DIR / input_name
        input_fasta_path.write_text(input_text)

    log_input = mo.md(
        f"**Data root:** `{ROOT}`  \n"
        f"**Dataset:** `{DATASET_DIR}`"
        + (f"  \n⚠️ `force_rebuild` is **on** — this dataset's artifacts were cleared and will be rebuilt."
           if force_rebuild.value else "")
        + f"  \n**Input:** {len(genome_accs)} accession(s) from *{genome_src}* · mode `{input_mode}`"
        + (f"  \nfirst accessions: `{', '.join(genome_accs[:5])}{' …' if len(genome_accs) > 5 else ''}`"
           if genome_accs else "")
    )

    return (
        ACC_RE,
        genome_accs,
        genome_set,
        input_fasta_path,
        input_mode,
        log_input,
    )


@app.cell
def acquire_genomes(
    ACC_RE,
    ASSEMBLY_REPORT,
    DATASET_DIR,
    GENOMES_DIR,
    GENOMES_FNA,
    IN_DIR,
    genome_accs,
    input_fasta_path,
    input_mode,
    mo,
    shutil,
    subprocess,
    zipfile,
):

    _acq_log = []

    def _split_genome_fasta(path, out_dir):
        """Split a multi-record nucleotide FASTA into <ACC>/<ACC>.fna."""
        cur, buf, n = None, [], 0

        def _flush(acc, lines):
            if not acc:
                return
            d = out_dir / acc
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{acc}.fna").write_text("\n".join(lines) + "\n")

        for line in path.read_text().splitlines():
            if line.startswith(">"):
                _flush(cur, buf)
                m = ACC_RE.search(line)
                cur = m.group(0) if m else line[1:].split("|")[0].split()[0]
                buf = [line]
                n += 1
            else:
                buf.append(line)
        _flush(cur, buf)
        shutil.copy(path, GENOMES_FNA)
        return n

    def _download_genomes(accs):
        if not accs:
            return
        accs_file = IN_DIR / "accessions.txt"
        accs_file.write_text("\n".join(accs) + "\n")
        zip_path = DATASET_DIR / "ncbi_dataset.zip"
        cmd = ["datasets", "download", "genome", "accession",
               "--inputfile", str(accs_file), "--include", "genome,gff3,protein",
               "--filename", str(zip_path), "--no-progressbar"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        _acq_log.append(f"datasets download: rc={r.returncode}, {len(accs)} accession(s)")
        if r.returncode != 0:
            _acq_log.append((r.stderr or r.stdout or "").strip()[-600:])
            return
        tmp = DATASET_DIR / "_download_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        data = tmp / "ncbi_dataset" / "data"
        n = 0
        if data.exists():
            for entry in data.iterdir():
                if entry.is_dir():
                    dst = GENOMES_DIR / entry.name
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.move(str(entry), str(dst))
                    n += 1
                elif entry.name == "assembly_data_report.jsonl":
                    shutil.copy(str(entry), str(ASSEMBLY_REPORT))
        shutil.rmtree(tmp, ignore_errors=True)
        zip_path.unlink(missing_ok=True)
        _acq_log.append(f"extracted {n} genome folder(s)")

    if input_mode == "accessions":
        _missing = [a for a in genome_accs if not (GENOMES_DIR / a).exists()]
        _download_genomes(_missing)
    elif input_mode == "genomes" and input_fasta_path is not None:
        _n = _split_genome_fasta(input_fasta_path, GENOMES_DIR)
        _acq_log.append(f"split input FASTA into {_n} genome folder(s)")
    elif input_mode == "proteins":
        _acq_log.append("protein FASTA input: genome download skipped")

    if not GENOMES_FNA.exists() and GENOMES_DIR.exists():
        with open(GENOMES_FNA, "w") as _out:
            for _acc in genome_accs:
                for _f in sorted((GENOMES_DIR / _acc).glob("*.fna")):
                    _out.write(_f.read_text())
        if GENOMES_FNA.exists():
            _acq_log.append(f"concatenated genomes.fna ({GENOMES_FNA.stat().st_size / 1e6:.0f} MB)")

    log_acq = (mo.md("**Acquisition** — "
          + ("; ".join(_acq_log) if _acq_log else "all genomes already present; nothing to download")))
    return (log_acq,)


@app.cell
def genome_metadata(
    ASSEMBLY_REPORT,
    GENOMES_DIR,
    GENOMES_TSV,
    IN_DIR,
    genome_accs,
    json,
    mo,
    pl,
    subprocess,
):

    def _norm_report(d):
        if "reports" in d:
            d = d["reports"][0]
        ai = d.get("assemblyInfo") or d.get("assembly_info") or {}
        st = d.get("assemblyStats") or d.get("assembly_stats") or {}
        org = d.get("organism") or {}
        infra = org.get("infraspecificNames") or org.get("infraspecific_names") or {}
        src = str(d.get("sourceDatabase") or d.get("source_database") or "")
        src = "RefSeq" if "REFSEQ" in src.upper() else ("GenBank" if "GENBANK" in src.upper() else "")
        length = st.get("totalSequenceLength") or st.get("total_sequence_length") or 0
        return {
            "accession": d.get("currentAccession") or d.get("current_accession") or d.get("accession"),
            "source": src,
            "level": ai.get("assemblyLevel") or ai.get("assembly_level") or "",
            "contigs": st.get("numberOfContigs") or st.get("number_of_contigs"),
            "length": int(length) if str(length).isdigit() else None,
            "species": org.get("organismName") or org.get("organism_name") or "",
            "strain": infra.get("strain") or "",
            "paired": d.get("pairedAccession") or d.get("paired_accession"),
        }

    def _report_rows(path):
        rows = []
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    try:
                        rows.append(_norm_report(json.loads(line)))
                    except Exception:
                        pass
        return rows

    def _summary_rows(accs):
        if not accs:
            return []
        accs_file = IN_DIR / "accessions.txt"
        if not accs_file.exists():
            accs_file.write_text("\n".join(accs) + "\n")
        r = subprocess.run(
            ["datasets", "summary", "genome", "accession",
             "--inputfile", str(accs_file), "--as-json-lines"],
            capture_output=True, text=True)
        if r.returncode != 0:
            return []
        rows = []
        for line in r.stdout.splitlines():
            if line.strip():
                try:
                    rows.append(_norm_report(json.loads(line)))
                except Exception:
                    pass
        return rows

    _rows = _report_rows(ASSEMBLY_REPORT)
    if not _rows and genome_accs:
        _rows = _summary_rows(genome_accs)
        if _rows:
            ASSEMBLY_REPORT.write_text("\n".join(json.dumps(r) for r in _rows) + "\n")

    _by_acc = {r["accession"]: r for r in _rows if r.get("accession")}
    _meta_rows = []
    for _acc in genome_accs:
        _r = dict(_by_acc.get(_acc, {}))
        _r["accession"] = _acc
        _r.setdefault("source", "RefSeq" if _acc.startswith("GCF") else "GenBank")
        _r.setdefault("level", "")
        _r.setdefault("contigs", None)
        _r.setdefault("length", None)
        _r.setdefault("species", "")
        _r.setdefault("strain", "")
        _d = GENOMES_DIR / _acc
        _fna = next(iter(sorted(_d.glob("*.fna"))), None) if _d.exists() else None
        _r["path"] = str(_fna) if _fna else ""
        _r["is_tiny"] = bool(_r.get("length") and _r["length"] < 1_000_000)
        _r.pop("paired", None)
        _meta_rows.append(_r)

    _schema = {"accession": pl.Utf8, "source": pl.Utf8, "level": pl.Utf8,
               "contigs": pl.Int64, "length": pl.Int64, "species": pl.Utf8,
               "strain": pl.Utf8, "path": pl.Utf8, "is_tiny": pl.Boolean}
    meta_df = pl.DataFrame(_meta_rows, schema=_schema) if _meta_rows else pl.DataFrame(schema=_schema)
    if meta_df.height:
        meta_df = (meta_df
                   .with_columns(pl.col("contigs").cast(pl.Int64, strict=False),
                                 pl.col("length").cast(pl.Int64, strict=False),
                                 pl.col("strain").fill_null(""),
                                 pl.col("is_tiny").fill_null(False))
                   .sort("accession"))
        _sp, _st = pl.col("species"), pl.col("strain")
        for _acc, (_s, _t) in {"GCF_020683165.1": ("Xanthobacter flavus", "GJ10"),
                               "GCF_060823235.1": ("Xanthobacter flavus", "GJ10")}.items():
            _sp = pl.when(pl.col("accession") == _acc).then(pl.lit(_s)).otherwise(_sp)
            _st = pl.when(pl.col("accession") == _acc).then(pl.lit(_t)).otherwise(_st)
        meta_df = meta_df.with_columns(_sp.alias("species"), _st.alias("strain"))
    meta_df.write_csv(GENOMES_TSV, separator="\t")

    log_meta = mo.md(
        f"**{meta_df.height} genomes** parsed from "
        f"{'the NCBI assembly report' if _rows else 'accessions only'} · "
        f"{meta_df.filter(pl.col('source') == 'RefSeq').height if meta_df.height else 0} RefSeq / "
        f"{meta_df.filter(pl.col('source') == 'GenBank').height if meta_df.height else 0} GenBank · "
        f"{meta_df.filter(pl.col('is_tiny')).height if meta_df.height else 0} tiny · "
        f"written to `{GENOMES_TSV}`.")

    genome_table = mo.ui.table(meta_df, selection=None, pagination=True, page_size=10)
    return genome_table, log_meta, meta_df


@app.cell
def _(genome_table, mo):
    mo.md(f"""
    ## Imported Genome Set:

    {genome_table}

    """)

    return


@app.cell
def disp_helpers(genome_accs, meta_df):
    meta_rows = {r["accession"]: r for r in meta_df.to_dicts()}

    def disp_name(acc):
        r = meta_rows.get(acc, {})
        sp = (r.get("species") or "?").replace("Xanthobacter", "X.")
        st = r.get("strain")
        st = "" if (not st or st == "missing") else f" {st}"
        return f"{sp}{st}"

    def disp_label(acc):
        return f"{disp_name(acc)}  ({acc})"

    disp_of = {a: disp_name(a) for a in genome_accs}
    return disp_label, disp_name, meta_rows


@app.cell(hide_code=True)
def meta_md(PROTEIN_FAA, dataset_slug, mo):

    _prot_dl = (
        mo.download(data=PROTEIN_FAA.read_bytes(),
                    filename=f"{dataset_slug.value}_proteins.faa",
                    mimetype="text/plain",
                    label=f"⬇ {PROTEIN_FAA.name} ({PROTEIN_FAA.stat().st_size / 1e6:.0f} MB)")
        if PROTEIN_FAA.exists() else mo.md("*`proteins.faa` not built yet.*"))

    mo.md(rf"""
    ### 2. Compile databases for PSI-BLAST and HMMER search

    Referencing the `meta/genomes.tsv` at the data root (built above, or created from scratch), the genome metadata is parsed through and depicted in the table in the previous step.

    The protein sequence database is saved to `{PROTEIN_FAA}`, also downloadable below. These files are used as a searchable database for both PSI-BLAST and HMMER. _Note: Genomes that do not have existing gene annotations are completed using pyrodigal, a python library that predicts ORFs._

    {_prot_dl}
    """)

    return


@app.cell
def db_proteins(
    GENOMES_DIR,
    PROTEIN_FAA,
    SeqIO,
    force_rebuild,
    genome_accs,
    input_fasta_path,
    input_mode,
    mo,
    shutil,
):

    def _build_proteins(log):
        if input_mode == "proteins" and input_fasta_path is not None:
            shutil.copy(str(input_fasta_path), str(PROTEIN_FAA))
            log.append(f"proteins from input FASTA ({PROTEIN_FAA.stat().st_size / 1e6:.1f} MB)")
            return
        if PROTEIN_FAA.exists() and PROTEIN_FAA.stat().st_size > 0 and not force_rebuild.value:
            log.append(f"{PROTEIN_FAA.name} cached ({PROTEIN_FAA.stat().st_size / 1e6:.1f} MB)")
            return
        import pyrodigal
        n_ncbi = n_pred = 0
        with open(PROTEIN_FAA, "w") as out:
            for acc in genome_accs:
                gdir = GENOMES_DIR / acc
                pfaa = gdir / "protein.faa"
                if pfaa.exists():
                    for rec in SeqIO.parse(str(pfaa), "fasta"):
                        desc = rec.description[len(rec.id):].strip()
                        out.write(f">{acc}|{rec.id} {desc}\n{rec.seq}\n")
                    n_ncbi += 1
                    continue
                fna = next(iter(sorted(gdir.glob("*.fna"))), None) if gdir.exists() else None
                if fna is None:
                    continue
                seq = "".join(l.strip() for l in fna.read_text().splitlines() if not l.startswith(">"))
                finder = pyrodigal.GeneFinder(meta=True)
                for i, gene in enumerate(finder.find_genes(seq.encode()), 1):
                    out.write(f">{acc}|{acc}_PRED_{i} predicted protein\n{gene.translate()}\n")
                n_pred += 1
        log.append(f"{n_ncbi} NCBI + {n_pred} pyrodigal-predicted proteomes → {PROTEIN_FAA.name}")

    _prot_log = []
    _build_proteins(_prot_log)
    log_proteins = (mo.md("**Proteins** — " + "; ".join(_prot_log)))
    return (log_proteins,)


@app.cell
def db_blastdb(BLAST_DB, PROTEIN_FAA, Path, force_rebuild, mo, subprocess):

    def _build_blastdb(log):
        idx = Path(str(BLAST_DB) + ".pin")
        if idx.exists() and idx.stat().st_size > 0 and not force_rebuild.value:
            log.append("BLAST DB cached")
            return
        if not PROTEIN_FAA.exists():
            log.append("⚠️ proteins.faa missing — skipped")
            return
        r = subprocess.run(["makeblastdb", "-in", str(PROTEIN_FAA), "-dbtype", "prot",
                            "-out", str(BLAST_DB)], capture_output=True, text=True)
        log.append(f"makeblastdb rc={r.returncode}"
                   + ("" if r.returncode == 0 else f" — {(r.stderr or '')[-200:]}"))

    _blast_log = []
    _build_blastdb(_blast_log)
    log_blast = (mo.md("**BLAST DB** — " + "; ".join(_blast_log)))
    return (log_blast,)


@app.cell
def db_hmms_seeds(HMM_DIR, QUERY_DIR, force_rebuild, gzip, mo, subprocess):

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

    def _fetch_hmms(log):
        need = [pf for pf in FAMILY_TABLE if force_rebuild.value or not (HMM_DIR / f"{pf}.hmm").exists()]
        if not need:
            log.append("all Pfam HMMs cached")
            return
        for pf in need:
            url = f"https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam/{pf}/?annotation=hmm"
            r = subprocess.run(["curl", "-sL", url], capture_output=True)
            data = r.stdout or b""
            if data[:2] == b"\x1f\x8b":
                try:
                    data = gzip.decompress(data)
                except OSError:
                    pass
            if data.startswith(b"HMMER3"):
                (HMM_DIR / f"{pf}.hmm").write_bytes(data)
                log.append(f"{pf} ✅")
            else:
                log.append(f"{pf} ⚠️ fetch failed")

    def _fetch_seeds(log):
        groups = {"all_3_ssap_queries.faa": SSAP_SEEDS, "exo_seeds.faa": EXO_SEEDS}
        for fname, accs in groups.items():
            dst = QUERY_DIR / fname
            if dst.exists() and not force_rebuild.value:
                continue
            recs = []
            for acc in accs:
                r = subprocess.run(["curl", "-sL",
                                    f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"],
                                   capture_output=True, text=True)
                if r.stdout.startswith(">"):
                    recs.append(r.stdout.strip())
            if recs:
                dst.write_text("\n".join(recs) + "\n")
                log.append(f"{fname}: {len(recs)} seed(s)")

    _db_log = []
    _fetch_hmms(_db_log)
    _fetch_seeds(_db_log)
    log_refs = mo.md("**Reference data** — " + ("; ".join(_db_log) if _db_log else "cached")
                     + f" · seeds in `{QUERY_DIR.name}/`.")
    pfam_table = mo.md("| Pfam | Name | Role |\n|------|------|------|\n"
         + "\n".join(f"| {pf} | {n} | {r} |" for pf, (n, r) in FAMILY_TABLE.items()))
    return FAMILY_TABLE, SSAP_SEEDS, log_refs, pfam_table


@app.cell(hide_code=True)
def step3_md(mo, pfam_table):
    mo.md(rf"""
    ### 3. Search for recombinases and exonucleases using PSI-BLAST and HMMER

    Once these databases are prepared, they can be searched against using PSI-BLAST and HMMER. These use "seeds" made in **Pfam/Interpro** such as common recombinases like lambda $\beta$ and RecT to search for distant relatives.

    #### **Seeds from Pfam**
    The following table lists the seeds that will be searched against the genomes by default. The first four represent different superfamilies of recombinases, responsible for annealing ssDNA together for homologous recombination. YgaJ is a family of 5-3' exonucleases, and Gam is an additional phage-encoded protein that can prevent host-nucleases from degrading linear DNA used for recombineering. These proteins can be **individually searched for** and then **analyzed for co-location** in a given genome to identify potential operons.

    {pfam_table}

    #### **Seed Recombinases for PSI-BLAST**
    From literature, the following recombinases are commonly integrated into new recombineering systems:

    | Seed | Accession | Source |
    |------|-----------|--------|
    | Redβ | P03698 | phage λ (Bet) |
    | RecT | P33228 | *E. coli* |
    | Pluβ | Q7N2Y8 | *Photorhabdus laumondii* |

    PSI-BLAST uses one recombinase for a similarity search, so each of these seeds can be separately used to search for recombinases in the _Xanthobacter_ genomes. This differs from HMMER, which requires a Multiple Sequence Alignment (MSA) of proteins to search for matches.
    """)
    return


@app.cell
def hmm_config(FAMILY_TABLE, HMM_DIR, PROTEIN_FAA, mo):
    HMM_FAMILIES = {pf: name for pf, (name, role) in FAMILY_TABLE.items()}
    SSAP_FAMILIES = {name for pf, (name, role) in FAMILY_TABLE.items() if role == "SSAP"}
    EXO_FAMILIES = {name for pf, (name, role) in FAMILY_TABLE.items() if role == "EXO"}
    ACCESSORY_FAMILIES = {name for pf, (name, role) in FAMILY_TABLE.items() if role == "accessory"}

    HMM_EVALUE_DEFAULT = -5  # log10 i-E; every family gets its own cutoff

    hmm_evalue_exps = mo.ui.dictionary({
        pf: mo.ui.slider(start=-50, stop=1, step=1, value=HMM_EVALUE_DEFAULT,
                         label=f"{HMM_FAMILIES[pf]} · {pf}", show_value=True)
        for pf in HMM_FAMILIES
    })
    hmm_run_button = mo.ui.run_button(label="⟳ Re-run hmmsearch (Pfam profiles)")


    def hmm_cutoffs(exp_by_pf):
        """Pfam accession -> i-E-value cutoff, from the per-family sliders."""
        return {pf: 10.0 ** exp for pf, exp in exp_by_pf.value.items()}


    def hmm_family_cutoffs(exp_by_pf):
        """Pfam family name -> i-E-value cutoff, from the per-family sliders."""
        return {HMM_FAMILIES[pf]: 10.0 ** exp for pf, exp in exp_by_pf.value.items()}


    _sliders = list(hmm_evalue_exps.elements.values())
    mo.md(f"""
    #### **3a. Searching with HMMER**
    For each Pfam family, HMMER searches the **Xanthobacter** protein database. The threshold for valid hits can be fine-tuned for each family below:

    **Files saved to:**
    \n *HMM Directory:* {HMM_DIR} {'✅' if HMM_DIR.exists() else '⚠️ missing'}
    \n *Protein Directory:* {PROTEIN_FAA.name} {'✅' if PROTEIN_FAA.exists() else '⚠️ missing'}

    {mo.vstack([mo.hstack(_sliders[i:i + 2], justify="start", gap=2)
                for i in range(0, len(_sliders), 2)])}

    {mo.hstack([hmm_run_button], justify="start", gap=2)}


    """)

    return (
        ACCESSORY_FAMILIES,
        EXO_FAMILIES,
        HMM_FAMILIES,
        SSAP_FAMILIES,
        hmm_cutoffs,
        hmm_evalue_exps,
        hmm_family_cutoffs,
        hmm_run_button,
    )


@app.cell
def hmm_load(
    HMM_DIR,
    HMM_FAMILIES,
    PROTEIN_FAA,
    RESULTS_DIR,
    genome_set,
    hmm_cutoffs,
    hmm_evalue_exps,
    hmm_family_cutoffs,
    hmm_run_button,
    json,
    mo,
    pl,
    subprocess,
):
    def _parse_domtbl(path, fam):
        out = []
        with open(path) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                c = line.split()
                if len(c) < 23:
                    continue
                genome, _, protein = c[0].partition("|")
                out.append({
                    "family": fam, "genome": genome, "protein": protein or c[0],
                    "i_evalue": float(c[12]), "dom_score": float(c[13]),
                    "tlen": int(c[2]),
                    "hmm_from": int(c[15]), "hmm_to": int(c[16]),
                    "ali_from": int(c[17]), "ali_to": int(c[18]),
                    "desc": " ".join(c[22:]),
                })
        return out


    def _load_hmm(do_rerun, thresholds):
        """Run hmmsearch once per family at that family's cutoff, then parse domtbls.

        A cached domtbl is reused when it was reported at least as permissively as the
        current cutoff, so tightening a cutoff re-filters for free while loosening one
        re-runs only the affected family. `hmmsearch_cutoffs.json` records the reporting
        cutoff of each cached search (absent = hmmsearch's default of 10).
        """
        log = []
        state_path = RESULTS_DIR / "hmmsearch_cutoffs.json"
        try:
            reported = json.loads(state_path.read_text()) if state_path.exists() else {}
        except (OSError, ValueError):
            reported = {}
        ready = PROTEIN_FAA.exists() and HMM_DIR.exists()
        used, ran_any = {}, False
        for pf, thr in thresholds.items():
            domtbl = RESULTS_DIR / f"hmmsearch_{pf}.domtbl"
            last = float(reported.get(pf, 10.0))
            if not ready:
                log.append(f"{pf}: ⚠️ HMM or protein DB missing — cached results only")
                used[pf] = last
                continue
            if not (do_rerun or not domtbl.exists() or thr > last):
                used[pf] = last
                continue
            r = subprocess.run(
                ["hmmsearch", "-o", str(RESULTS_DIR / f"hmmsearch_{pf}.out"),
                 "--domtblout", str(domtbl), "-E", f"{thr:.1e}", "--domE", f"{thr:.1e}",
                 str(HMM_DIR / f"{pf}.hmm"), str(PROTEIN_FAA)],
                capture_output=True, text=True)
            if r.returncode == 0:
                used[pf] = last = thr
                ran_any = True
                log.append(f"{pf}: rc=0 at i-E ≤ {thr:.0e}")
            else:
                used[pf] = last
                log.append(f"{pf}: rc={r.returncode} — {(r.stderr or '').strip()[-160:]}")
        if ran_any:
            try:
                state_path.write_text(json.dumps(used, indent=2, sort_keys=True))
            except OSError as exc:
                log.append(f"⚠️ cutoff cache not written: {exc}")
        rows = []
        for pf, fam in HMM_FAMILIES.items():
            p = RESULTS_DIR / f"hmmsearch_{pf}.domtbl"
            if p.exists():
                rows += _parse_domtbl(p, fam)
        schema = {"family": pl.Utf8, "genome": pl.Utf8, "protein": pl.Utf8,
                  "i_evalue": pl.Float64, "dom_score": pl.Float64, "tlen": pl.Int64,
                  "hmm_from": pl.Int64, "hmm_to": pl.Int64, "ali_from": pl.Int64,
                  "ali_to": pl.Int64, "desc": pl.Utf8}
        df = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
        if df.height:
            df = df.filter(pl.col("genome").is_in(list(genome_set)))
        return df, log


    _fam_cutoffs = hmm_family_cutoffs(hmm_evalue_exps)
    hmm_hits_all, _hmm_log = _load_hmm(hmm_run_button.value, hmm_cutoffs(hmm_evalue_exps))
    log_hmm_load = mo.md(
        f"Parsed **{hmm_hits_all.height} domain rows** across "
        f"{hmm_hits_all['genome'].n_unique() if hmm_hits_all.height else 0} genomes at "
        f"per-family i-E cutoffs ("
        + ", ".join(f"{fam} ≤ {thr:.0e}" for fam, thr in sorted(_fam_cutoffs.items()))
        + ")" + (f"  \nRe-run: {', '.join(_hmm_log)}" if _hmm_log else ""))

    return hmm_hits_all, log_hmm_load


@app.cell
def _(collections, meta_rows, pl):

    def combine_role_hits(hmm_df, psi_df, families, psi_method):
        """One row per (genome, protein) with HMM and/or PSI-BLAST evidence merged."""
        agg = collections.defaultdict(
            lambda: {"hmm_families": set(), "hmm_evalue": None,
                     "psi_seeds": set(), "psi_evalue": None})
        if hmm_df is not None and hmm_df.height:
            for r in hmm_df.filter(pl.col("family").is_in(list(families))).iter_rows(named=True):
                k = (r["genome"], r["protein"])
                agg[k]["hmm_families"].add(r["family"])
                e = r["i_evalue"]
                agg[k]["hmm_evalue"] = e if agg[k]["hmm_evalue"] is None else min(agg[k]["hmm_evalue"], e)
        if psi_df is not None and psi_df.height:
            for r in psi_df.iter_rows(named=True):
                k = (r["genome"], r["protein"])
                agg[k]["psi_seeds"].add(r["seed"])
                e = r["evalue"]
                agg[k]["psi_evalue"] = e if agg[k]["psi_evalue"] is None else min(agg[k]["psi_evalue"], e)
        rows = []
        for (genome, protein), d in agg.items():
            methods = []
            if d["hmm_families"]:
                methods.append("HMM")
            if d["psi_seeds"]:
                methods.append(psi_method)
            evs = [x for x in (d["hmm_evalue"], d["psi_evalue"]) if x is not None]
            rows.append({
                "genome": genome, "protein": protein,
                "species": meta_rows.get(genome, {}).get("species", ""),
                "hmm_families": ", ".join(sorted(d["hmm_families"])) or "—",
                "psiblast_seeds": ", ".join(sorted(d["psi_seeds"])) or "—",
                "method": "+".join(methods),
                "best_evalue": min(evs) if evs else None,
                "hmm_evalue": d["hmm_evalue"], "psi_evalue": d["psi_evalue"],
            })
        if not rows:
            return pl.DataFrame(schema={"genome": pl.Utf8, "protein": pl.Utf8, "species": pl.Utf8,
                                        "hmm_families": pl.Utf8, "psiblast_seeds": pl.Utf8,
                                        "method": pl.Utf8, "best_evalue": pl.Float64,
                                        "hmm_evalue": pl.Float64, "psi_evalue": pl.Float64})
        return pl.DataFrame(rows).sort("best_evalue")

    # mo.md("*`combine_role_hits` merges HMMER and PSI-BLAST evidence per protein.*")
    return (combine_role_hits,)


@app.cell
def hmm_filter(hmm_evalue_exps, hmm_family_cutoffs, hmm_hits_all, mo, pl):
    _cutoffs = hmm_family_cutoffs(hmm_evalue_exps)
    if hmm_hits_all.height:
        hmm_hits_df = (
            hmm_hits_all.with_columns(
                pl.col("family").replace_strict(_cutoffs, default=0.0).alias("_i_e_cutoff"))
            .filter(pl.col("i_evalue") <= pl.col("_i_e_cutoff"))
            .sort("i_evalue")
            .unique(subset=["genome", "protein", "family"], keep="first")
            .sort(["family", "i_evalue"])
        )
    else:
        hmm_hits_df = hmm_hits_all

    log_hmm = (mo.md(
        f"**{hmm_hits_df.height} called profile hits** (best domain per protein/family) at "
        f"per-family i-E cutoffs ("
        + ", ".join(f"{fam} ≤ {thr:.0e}" for fam, thr in sorted(_cutoffs.items()))
        + ") in "
        f"{hmm_hits_df['genome'].n_unique() if hmm_hits_df.height else 0} genomes — "
        f"the recombinase table follows below; the exonuclease / inhibitor cutoffs are separate."))

    return hmm_hits_df, log_hmm


@app.cell
def hmm_recomb_table(
    SSAP_FAMILIES,
    hmm_evalue_exps,
    hmm_family_cutoffs,
    hmm_hits_df,
    meta_df,
    mo,
    pl,
):
    _cutoffs = hmm_family_cutoffs(hmm_evalue_exps)
    _hmm_rec = (hmm_hits_df.filter(pl.col("family").is_in(list(SSAP_FAMILIES)))
                .join(meta_df.select([pl.col("accession").alias("genome"), "species"]),
                      on="genome", how="left")
                .select(["species", "genome", "protein", "family", "i_evalue", "dom_score",
                         "ali_from", "ali_to", "desc"])
                .sort("i_evalue"))
    sum_hmm_rec = mo.md(
        f"**{_hmm_rec.height} recombinase hits** from HMMER at per-family i-E cutoffs ("
        + ", ".join(f"{fam} ≤ {_cutoffs[fam]:.0e}" for fam in sorted(SSAP_FAMILIES))
        + f") · {_hmm_rec['protein'].n_unique() if _hmm_rec.height else 0} unique proteins across "
        f"{_hmm_rec['genome'].n_unique() if _hmm_rec.height else 0} genomes. "
        f"Profile families: {', '.join(sorted(SSAP_FAMILIES))}.")
    aggregate_recombinase_table = mo.ui.table(_hmm_rec, selection=None, pagination=True, page_size=15)

    mo.md(f"""
    ### Results from HMMER across all Pfam Families:

    {aggregate_recombinase_table}

    """)
    return (sum_hmm_rec,)


@app.cell
def psi_config(BLAST_DB, Path, QUERY_DIR, RESULTS_DIR, SSAP_SEEDS, mo):
    SEED_FAA = QUERY_DIR / "all_3_ssap_queries.faa"
    SEED_DIR = QUERY_DIR / "ssap_seeds"                 # per-seed query FASTAs (split on demand)
    PSIBLAST_TSV = RESULTS_DIR / "psiblast_ssap.tsv"    # legacy combined table (split on demand)
    PSI_ITERATIONS = 3
    PSI_EVALUE_DEFAULT = -3  # log10 E; every seed gets its own cutoff


    def psi_tsv(acc):
        """Per-seed PSI-BLAST output table."""
        return RESULTS_DIR / f"psiblast_{acc}.tsv"


    psi_evalue_exps = mo.ui.dictionary({
        acc: mo.ui.slider(start=-50, stop=1, step=1, value=PSI_EVALUE_DEFAULT,
                          label=f"{name} · {acc}", show_value=True)
        for acc, name in SSAP_SEEDS.items()
    })
    psi_run_button = mo.ui.run_button(label="⟳ Re-run PSI-BLAST (per-seed cutoffs, 3 iters)")


    def psi_cutoffs(exp_by_seed):
        """Seed accession -> E-value cutoff, from the per-seed sliders."""
        return {acc: 10.0 ** exp for acc, exp in exp_by_seed.value.items()}


    _sliders = list(psi_evalue_exps.elements.values())
    _status = " · ".join(
        f"{name} `{psi_tsv(acc).name}` {'✅' if psi_tsv(acc).exists() else '⚠️'}"
        for acc, name in SSAP_SEEDS.items())

    psi_blast_parameters = mo.vstack([
        mo.vstack([mo.hstack(_sliders[i:i + 2], justify="start", gap=2)
                   for i in range(0, len(_sliders), 2)]),
        mo.hstack([psi_run_button], justify="start", gap=2),
        mo.md(f"db: `{BLAST_DB.name}` {'✅' if Path(str(BLAST_DB) + '.pin').exists() else '⚠️'} · "
              f"per-seed tables — {_status}"),
    ])

    mo.md(f"""
    #### 3b. **Searching with PSI-BLAST**
    Using the three recombinases listed above, PSI-BLAST is used to search for similar sequences in the **Xanthobacter** protein database.

    **Files saved to:**
    \n {_status}

    **Modify Cutoffs**
    {psi_blast_parameters}
    """)
    return (
        PSIBLAST_TSV,
        PSI_EVALUE_DEFAULT,
        PSI_ITERATIONS,
        SEED_DIR,
        SEED_FAA,
        psi_cutoffs,
        psi_evalue_exps,
        psi_run_button,
        psi_tsv,
    )


@app.cell
def psi_load(
    BLAST_DB,
    PSIBLAST_TSV,
    PSI_EVALUE_DEFAULT,
    PSI_ITERATIONS,
    RESULTS_DIR,
    SEED_DIR,
    SEED_FAA,
    SSAP_SEEDS,
    SeqIO,
    genome_set,
    json,
    mo,
    parse_psiblast_tsv,
    pl,
    psi_cutoffs,
    psi_evalue_exps,
    psi_run_button,
    psi_tsv,
    seed_id,
    subprocess,
):
    def _seed_query_files():
        """Per-seed query FASTAs, split out of the combined SSAP seed file."""
        SEED_DIR.mkdir(parents=True, exist_ok=True)
        if SEED_FAA.exists():
            recs = {}
            for rec in SeqIO.parse(str(SEED_FAA), "fasta"):
                if seed_id(rec.id) in SSAP_SEEDS:
                    recs.setdefault(seed_id(rec.id), []).append(rec)
            for acc, rs in recs.items():
                p = SEED_DIR / f"{acc}.faa"
                if not p.exists():
                    SeqIO.write(rs, str(p), "fasta")
        return {acc: SEED_DIR / f"{acc}.faa" for acc in SSAP_SEEDS
                if (SEED_DIR / f"{acc}.faa").exists()}


    def _psiblast_state(seed_tsvs):
        """Cutoff recorded for each cached per-seed table.

        On first use the legacy single-run `psiblast_ssap.tsv` is split into
        per-seed tables: PSI-BLAST builds one profile per query, so a combined
        multi-query run is equivalent to separate runs at the same cutoff.
        """
        state_path = RESULTS_DIR / "psiblast_cutoffs.json"
        try:
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
        except (OSError, ValueError):
            state = {}
        if state or not PSIBLAST_TSV.exists():
            return state, state_path
        buckets = {acc: [] for acc in SSAP_SEEDS}
        try:
            lines = PSIBLAST_TSV.read_text().splitlines()
        except OSError:
            return state, state_path
        for line in lines:
            if not line.strip() or line.startswith(("Search", "#")):
                continue
            acc = seed_id(line.split("\t")[0])
            if acc in buckets:
                buckets[acc].append(line)
        if not any(buckets.values()):
            return state, state_path
        for acc, rows in buckets.items():
            if not seed_tsvs[acc].exists():
                seed_tsvs[acc].write_text("\n".join(rows) + ("\n" if rows else ""))
        state = {acc: 10.0 ** PSI_EVALUE_DEFAULT for acc in SSAP_SEEDS}
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
        return state, state_path


    def _load_psiblast(do_rerun, cutoffs):
        """Run PSI-BLAST once per seed at that seed's E cutoff, then parse tables.

        A cached table is reused when it was reported at least as permissively as
        the current cutoff, so tightening a cutoff re-filters for free while
        loosening one re-runs only that seed. The cutoff of each cached run is kept
        in `psiblast_cutoffs.json`.
        """
        log = []
        seed_tsvs = {acc: psi_tsv(acc) for acc in SSAP_SEEDS}
        queries = _seed_query_files()
        state, state_path = _psiblast_state(seed_tsvs)
        used, ran_any = {}, False
        for acc, thr in cutoffs.items():
            out = seed_tsvs[acc]
            last = float(state.get(acc, 10.0 ** PSI_EVALUE_DEFAULT))
            q = queries.get(acc)
            if q is None:
                log.append(f"{SSAP_SEEDS[acc]}: ⚠️ no query FASTA — cached results only")
                used[acc] = last
                continue
            if not (do_rerun or not out.exists() or thr > last):
                used[acc] = last
                continue
            r = subprocess.run(
                ["psiblast", "-query", str(q), "-db", str(BLAST_DB),
                 "-num_iterations", str(PSI_ITERATIONS),
                 "-evalue", f"{thr:.1e}", "-inclusion_ethresh", f"{thr:.1e}",
                 "-outfmt", "6 std qcovhsp", "-num_threads", "4", "-out", str(out)],
                capture_output=True, text=True)
            if r.returncode == 0:
                used[acc] = last = thr
                ran_any = True
                log.append(f"{SSAP_SEEDS[acc]}: rc=0 at E ≤ {thr:.0e}")
            else:
                used[acc] = last
                log.append(f"{SSAP_SEEDS[acc]}: rc={r.returncode} — {(r.stderr or '').strip()[-160:]}")
        if ran_any:
            try:
                state_path.write_text(json.dumps(used, indent=2, sort_keys=True))
            except OSError as exc:
                log.append(f"⚠️ cutoff cache not written: {exc}")
        rows = []
        for p in seed_tsvs.values():
            if p.exists():
                rows += parse_psiblast_tsv(p)
        schema = {"seed": pl.Utf8, "genome": pl.Utf8, "protein": pl.Utf8,
                  "pident": pl.Float64, "aln_len": pl.Int64, "evalue": pl.Float64,
                  "bitscore": pl.Float64, "qcovhsp": pl.Int64}
        df = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
        if df.height:
            df = df.filter(pl.col("genome").is_in(list(genome_set)))
        return df, log


    _cutoffs = psi_cutoffs(psi_evalue_exps)
    psi_hits_all, _psi_log = _load_psiblast(psi_run_button.value, _cutoffs)
    log_psi_load = mo.md(
        f"Parsed **{psi_hits_all.height} SSAP PSI-BLAST HSP rows** at per-seed E cutoffs ("
        + ", ".join(f"{SSAP_SEEDS[acc]} ≤ {thr:.0e}" for acc, thr in sorted(_cutoffs.items()))
        + ")" + (f"  \nRe-run: {', '.join(_psi_log)}" if _psi_log else ""))

    return log_psi_load, psi_hits_all


@app.cell
def psi_filter(
    SSAP_SEEDS,
    meta_df,
    mo,
    pl,
    psi_cutoffs,
    psi_evalue_exps,
    psi_hits_all,
):
    _cutoffs = psi_cutoffs(psi_evalue_exps)
    if psi_hits_all.height:
        psi_hits_df = (
            psi_hits_all.with_columns(
                pl.col("seed").replace_strict(_cutoffs, default=0.0).alias("_e_cutoff"))
            .filter(pl.col("evalue") <= pl.col("_e_cutoff"))
            .sort("evalue")
            .unique(subset=["seed", "genome", "protein"], keep="first")
            .sort(["seed", "evalue"])
        )
    else:
        psi_hits_df = psi_hits_all

    _psi_rec = (psi_hits_df
                .join(meta_df.select([pl.col("accession").alias("genome"), "species"]),
                      on="genome", how="left")
                .select(["species", "genome", "protein", "seed", "pident", "aln_len",
                         "qcovhsp", "evalue", "bitscore"])
                .sort("evalue"))
    sum_psi_rec = mo.md(
        f"**{_psi_rec.height} recombinase hits** from PSI-BLAST at per-seed E cutoffs ("
        + ", ".join(f"{SSAP_SEEDS[acc]} ≤ {thr:.0e}" for acc, thr in sorted(_cutoffs.items()))
        + f") · {_psi_rec['protein'].n_unique() if _psi_rec.height else 0} unique proteins across "
        f"{_psi_rec['genome'].n_unique() if _psi_rec.height else 0} genomes. "
        f"Seeds: {', '.join(sorted(SSAP_SEEDS.values()))}.")

    mo.ui.table(_psi_rec, selection=None, pagination=True, page_size=15)
    return psi_hits_df, sum_psi_rec


@app.cell
def recomb_hits_hdr(mo):
    mo.md(r"""
    ### 4. Deduplicating Recombinases based off of % Similarities

    After identifying recombinases from HMMER and PSI-BLAST, we can look at overlapping sequences between the search methods. Below is a table identifying each unique recombinase protein as well as the method (HMM, PSI-BLAST, or both) that identified them.
    """)
    return


@app.cell
def merged_recomb_table(
    SSAP_FAMILIES,
    combine_role_hits,
    hmm_hits_df,
    mo,
    pl,
    psi_hits_df,
):

    recombine_hits = combine_role_hits(hmm_hits_df, psi_hits_df, SSAP_FAMILIES, "PSI-BLAST")
    _n_hmm = recombine_hits.filter(pl.col("method").str.contains("HMM")).height if recombine_hits.height else 0
    _n_psi = recombine_hits.filter(pl.col("method").str.contains("PSI")).height if recombine_hits.height else 0
    sum_merged_rec = mo.md(
        f"**{recombine_hits.height} recombinase hits** merging HMMER and PSI-BLAST "
        f"({_n_hmm} with profile evidence, {_n_psi} with PSI-BLAST evidence, "
        f"{recombine_hits.filter(pl.col("method").str.contains("HMM") & pl.col("method").str.contains("PSI")).height} with both) across "
        f"{recombine_hits['genome'].n_unique() if recombine_hits.height else 0} genomes.")

    mo.md(f"""
    #### **4a. Unique Recombinases from Both HMM and PSI-BLAST:**

    {mo.ui.table(recombine_hits, selection=None, pagination=True, page_size=15)}

    There are **{recombine_hits.height}** unique recombinases.
    """)

    return recombine_hits, sum_merged_rec


@app.cell
def rec_hits_charts(alt, mo, pdf, pl, recombine_hits, result):
    if recombine_hits.height:
        _method_order = ["HMM", "PSI-BLAST", "Both"]
        _colors = alt.Scale(domain=_method_order, range=["#4c78a8", "#f58518", "#54a24b"])
        _rec = recombine_hits.with_columns(
            pl.when(pl.col("method").str.contains("HMM") & pl.col("method").str.contains("PSI"))
            .then(pl.lit("Both"))
            .when(pl.col("method").str.contains("HMM")).then(pl.lit("HMM"))
            .otherwise(pl.lit("PSI-BLAST")).alias("method_label"))

        _by_species = (_rec.group_by(["species", "method_label"])
                       .agg(pl.col("protein").n_unique().alias("unique")))
        _species_chart = (alt.Chart(pdf(_by_species))
                          .mark_bar()
                          .encode(
                              x=alt.X("unique:Q", title="unique recombinases"),
                              y=alt.Y("species:N", sort="-x", title=None),
                              color=alt.Color("method_label:N", title="method", scale=_colors),
                              tooltip=["species", "method_label", "unique"])
                          .properties(height=max(180, 20 * _by_species["species"].n_unique()),
                                      width=400, title="Unique Recombinases by Species"))

        _by_sf = (_rec.select(["protein", "method_label",
                               pl.col("hmm_families")
                                 .str.replace_all("—", "PSI-BLAST only (no Pfam)")
                                 .str.split(", ").alias("superfamily")])
                  .explode("superfamily", empty_as_null=True)
                  .group_by(["superfamily", "method_label"])
                  .agg(pl.col("protein").n_unique().alias("unique")))
        _sf_totals = (_by_sf.group_by("superfamily").agg(pl.col("unique").sum())
                      .sort("unique", descending=True))
        _sf_chart = (alt.Chart(pdf(_by_sf))
                     .mark_bar()
                     .encode(
                         x=alt.X("unique:Q", title="unique recombinases"),
                         y=alt.Y("superfamily:N", sort="-x", title=None),
                         color=alt.Color("method_label:N", title="method", scale=_colors),
                         tooltip=["superfamily", "method_label", "unique"])
                     .properties(height=max(140, 30 * _by_sf["superfamily"].n_unique()),
                                 width=400, title="Unique Recombinases by Superfamily"))

        sum_superfam = mo.md(
            f"**{recombine_hits.height} merged recombinase hits** "
            f"({recombine_hits['protein'].n_unique()} unique proteins) across "
            f"**{_sf_totals.height} superfamilies** — "
            + " · ".join(f"{r['superfamily']} {r['unique']}" for r in _sf_totals.iter_rows(named=True))
            + ". Hits found only by PSI-BLAST, with no Pfam profile, would appear as a "
              "`PSI-BLAST only (no Pfam)` bucket.")
        _out = mo.hstack([mo.ui.altair_chart(_sf_chart), mo.ui.altair_chart(_species_chart)],
                         align="start", gap=2, widths="equal")
    else:
        sum_superfam = mo.md("*No merged recombinase hits at the current thresholds.*")
        _out = sum_superfam
    result(_out)

    return (sum_superfam,)


@app.cell(hide_code=True)
def ssap_dedup_inputs(
    PROTEIN_FAA,
    SSAP_FAMILIES,
    SSAP_SEEDS,
    load_protein_seqs,
    pl,
    recombine_hits,
):
    # Recombinase-only catalog + sequences for the deduplication section. The full
    # catalog (`recomb_df` / `recomb_seqs`) is built after the exonuclease search, so
    # the similarity analysis uses this recombinase-only view instead; it matches the
    # SSAP rows of the full catalog exactly.
    _hmm_rows = (recombine_hits
                 .filter(pl.col("hmm_families") != "—")
                 .select(["genome", "protein",
                          pl.col("hmm_families").str.split(", ").alias("family")])
                 .explode("family", empty_as_null=True)
                 .filter(pl.col("family").is_in(list(SSAP_FAMILIES))))
    _psi_rows = (recombine_hits
                 .filter(pl.col("method").str.contains("PSI") & ~pl.col("method").str.contains("HMM"))
                 .select(["genome", "protein", "psiblast_seeds"])
                 .with_columns(
                     pl.col("psiblast_seeds")
                     .map_elements(lambda s: "SSAP(PSI:" + SSAP_SEEDS.get(
                         s.split(", ")[0], s.split(", ")[0]) + ")",
                         return_dtype=pl.Utf8)
                     .alias("family"))
                 .select(["genome", "protein", "family"]))
    ssap_catalog = (pl.concat([_hmm_rows, _psi_rows])
                    .with_columns(pl.lit("SSAP").alias("role"))
                    .unique()
                    .sort(["family", "genome", "protein"]))
    ssap_seqs = load_protein_seqs(PROTEIN_FAA, set(ssap_catalog["protein"]))

    return ssap_catalog, ssap_seqs


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### **4b. Evaluating recombinases using % similarity**
    Since recombinases are identified across a variety of *Xanthobacter* genomes, while their protein accessions are considered separate, their sequences may be close enough to each other that they can be considered the same sequence. This is evaluated using **% similarity**, which performs a **multiple sequence alignment (MSA)** consisting of every protein and compares each of their sequence. A **BLOSUM62 matrix** weights differences in sequences differently based off of the physical properties of the amino acids; for example, the change from leucine to isoleucine is less penalized compared to the change from isoleucine to glycine.
    """)
    return


@app.cell
def uniq_matrices(
    MSA_DIR,
    SSAP_FAMILIES,
    SeqIO,
    force_rebuild,
    io,
    mo,
    np,
    pl,
    re,
    ssap_catalog,
    ssap_seqs,
    subprocess,
    substitution_matrices,
):

    _B62 = substitution_matrices.load("BLOSUM62")

    def _b62(x, y):
        try:
            return float(_B62[x, y])
        except Exception:
            return 0.0

    def _recs_for(fam):
        out = []
        for r in ssap_catalog.filter(pl.col("family") == fam).iter_rows(named=True):
            s = ssap_seqs.get(r["protein"])
            if s:
                out.append((f"{r['genome']}|{r['protein']}", s, r["family"], r["role"]))
        return out

    def _fam_msa(recs):
        """MAFFT alignment of one family; returns (labels, %id, %sim, coverage)."""
        n = len(recs)
        labels = [r[0] for r in recs]
        if n == 0:
            z = np.zeros((0, 0)); return labels, z, z, z
        if n == 1:
            one = np.array([[100.0]]); return labels, one, one, np.array([[1.0]])
        fasta = "\n".join(f">{i}\n{s}" for i, (_, s, _, _) in enumerate(recs))
        r = subprocess.run(["mafft", "--quiet", "--auto", "-"], input=fasta,
                           capture_output=True, text=True)
        by = {int(a.id): str(a.seq) for a in SeqIO.parse(io.StringIO(r.stdout), "fasta")}
        lens = [len(s) for _, s, _, _ in recs]
        ident = np.full((n, n), np.nan); sim = np.full((n, n), np.nan); cov = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i, n):
                si, sj = by[i], by[j]
                both = same = pos = 0
                for x, y in zip(si, sj):
                    if x != "-" and y != "-":
                        both += 1
                        same += int(x == y)
                        pos += int(_b62(x, y) > 0)
                ident[i, j] = ident[j, i] = 100.0 * same / both if both else 0.0
                sim[i, j] = sim[j, i] = 100.0 * pos / both if both else 0.0
                m = min(lens[i], lens[j])
                cov[i, j] = cov[j, i] = both / m if m else 0.0
        return labels, ident, sim, cov

    def _is_recombinase_family(f):
        return (f in SSAP_FAMILIES) or f.startswith("SSAP(PSI:")

    def _build_all_msa():
        d = {}
        for fam in sorted(f for f in ssap_catalog["family"].unique().to_list()
                          if _is_recombinase_family(f)):
            recs = _recs_for(fam)
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", fam)
            cache = MSA_DIR / f"{safe}.npz"
            labels = [r[0] for r in recs]
            if cache.exists() and not force_rebuild.value:
                z = np.load(cache, allow_pickle=True)
                if set(map(str, z["labels"])) == set(labels):
                    d[fam] = (list(map(str, z["labels"])), z["ident"], z["sim"], z["cov"], recs)
                    continue
            labels, ident, sim, cov = _fam_msa(recs)
            d[fam] = (labels, ident, sim, cov, recs)
            np.savez(cache, labels=np.array(labels), ident=ident, sim=sim, cov=cov)
        return d

    unique_msa = _build_all_msa()
    log_msa = (mo.md("Per-family MAFFT alignment built for: "
          + ", ".join(f"`{f}` (n={len(v[0])})" for f, v in unique_msa.items())
          + ". Metrics: **% identity**, **BLOSUM62 % similarity**, **coverage**. Cached under `msa/`."))
    return log_msa, unique_msa


@app.cell
def uniq_controls(mo):

    dup_id_slider = mo.ui.slider(50, 100, step=1, value=90,
                                 label="Duplicate identity ≥ (%)", show_value=True)
    sys_id_slider = mo.ui.slider(30, 100, step=1, value=70,
                                 label="System/divergent identity ≥ (%)", show_value=True)
    cov_slider = mo.ui.slider(0.0, 1.0, step=0.05, value=0.8,
                              label="Min coverage", show_value=True)

    mo.md(f"""
    Different thresholds can be controlled to classify two sequences as duplicate (default >90%), divergent (<70%). Minimum coverage refers to how much of the full sequence is aligned with the other in the MSA.

    {mo.vstack([mo.md("**Deduplication Parameters**"), dup_id_slider, sys_id_slider, cov_slider])}

    """)

    return cov_slider, dup_id_slider, sys_id_slider


@app.cell
def uniq_clusters(
    collections,
    cov_slider,
    dup_id_slider,
    meta_rows,
    mo,
    pl,
    sys_id_slider,
    unique_msa,
):

    def single_linkage(labels, ident, cov, thr_id, thr_cov):
        n = len(labels)
        adj = collections.defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                if ident[i, j] >= thr_id and cov[i, j] >= thr_cov:
                    adj[i].add(j); adj[j].add(i)
        seen, comps = set(), []
        for i in range(n):
            if i in seen:
                continue
            stack, comp = [i], []
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x); comp.append(x); stack.extend(adj[x] - seen)
            comps.append(sorted(comp))
        return comps

    def rep_of(recs, idxs):
        def key(i):
            genome = recs[i][0].split("|")[0]; r = meta_rows.get(genome, {})
            return (0 if genome.startswith("GCF") else 1,
                    0 if r.get("level") == "Complete Genome" else 1,
                    r.get("contigs") or 10**9, -(r.get("length") or 0), -len(recs[i][1]))
        return min(idxs, key=key)

    def _build_unique(thr_dup, thr_sys, thr_cov):
        rows = []
        dup_cluster_of, sys_cluster_of = {}, {}
        for fam, (labels, ident, sim, cov, recs) in unique_msa.items():
            if not labels:
                continue
            dup = single_linkage(labels, ident, cov, thr_dup, thr_cov)
            sysc = single_linkage(labels, ident, cov, thr_sys, thr_cov)
            sysid = {i: gi for gi, comp in enumerate(sysc) for i in comp}
            for ci, comp in enumerate(dup, 1):
                for i in comp:
                    dup_cluster_of[labels[i]] = f"{fam}#{ci}"
            for i in range(len(labels)):
                sys_cluster_of[labels[i]] = f"{fam}#S{sysid[i] + 1}"
            dupinfo = [(comp, rep_of(recs, comp)) for comp in dup]
            sg_subtypes = collections.Counter(sysid[rep] for _, rep in dupinfo)
            for comp, rep in dupinfo:
                sg = sysid[rep]
                rows.append({
                    "family": fam, "role": recs[rep][3],
                    "representative": recs[rep][0],
                    "n_strain_dupes": len(comp),
                    "system_group": f"{fam}#{sg + 1}",
                    "subtypes_in_system": sg_subtypes[sg],
                    "is_divergent": sg_subtypes[sg] == 1,
                    "members": ", ".join(recs[i][0] for i in comp),
                })
        schema_cols = ["family", "role", "representative", "n_strain_dupes",
                       "system_group", "subtypes_in_system", "is_divergent", "members"]
        df = (pl.DataFrame(rows).sort(["family", "representative"])
              if rows else pl.DataFrame({c: [] for c in schema_cols}))
        return df, dup_cluster_of, sys_cluster_of

    unique_df, dup_cluster_of, sys_cluster_of = _build_unique(
        dup_id_slider.value, sys_id_slider.value, cov_slider.value)
    unique_rep_set = set(unique_df["representative"].to_list()) if unique_df.height else set()

    sum_unique = mo.md(
        f"**{unique_df.height} unique recombinases** "
        f"(strain dupes collapsed at ≥{dup_id_slider.value}% id / ≥{cov_slider.value} cov) · "
        f"**{unique_df.filter(pl.col('is_divergent')).height if unique_df.height else 0} divergent** "
        f"subtypes at ≥{sys_id_slider.value}% id.")

    mo.md(f"""
    *Deduplication Results*
    \n{sum_unique}

    *Summary of Grouped Recombinases:*
    {mo.ui.table(unique_df.select(
        ['system_group', 'family', 'role', 'representative', 'n_strain_dupes',
         'subtypes_in_system', 'is_divergent']),
        selection=None, pagination=True, page_size=15)}

    """)

    return (
        dup_cluster_of,
        rep_of,
        single_linkage,
        sum_unique,
        unique_df,
        unique_rep_set,
    )


@app.cell(hide_code=True)
def cmp_hdr(mo):
    mo.md(f"""
    **% Similarity Heatmaps**

    Evaluation generates pairwise similarity scores for each recombinase, which can be represented visually as a heatmap. This can help visually adjust parameters to determine reasonable cutoffs for groupings. Use the dropdown to compare based off of the recombinase family.
    """)
    return


@app.cell
def cmp_controls(mo, unique_msa):

    _fams = sorted(unique_msa.keys())
    cmp_family = mo.ui.dropdown(options=_fams, value=(_fams[0] if _fams else None),
                                label="Family")
    cmp_metric = mo.ui.radio(
        options=["BLOSUM62 % similarity", "MAFFT % identity", "MAFFT coverage"],
        value="BLOSUM62 % similarity", label="Metric", inline=True)
    cmp_order = mo.ui.radio(
        options=["Clustered (duplicate groups)", "Plain (catalog order)"],
        value="Clustered (duplicate groups)", label="Ordering", inline=True)
    mo.vstack([
        mo.hstack([cmp_family, cmp_metric], justify="start", gap=2),
        cmp_order,
    ])
    return cmp_family, cmp_metric, cmp_order


@app.cell
def cmp_compute(cmp_family, cmp_metric, mo, np, unique_msa):

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

    mo.md(f"**{cmp_metric.value}** over **{len(cmp_labels)} proteins** in "
          f"`{cmp_family.value}`"
          + ("" if cmp_labels else " — no sequences available."))
    return cmp_labels, cmp_matrix, cmp_unit


@app.cell
def cmp_heatmap(
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
        _labels = [l.split("|")[-1] for l in _ord]
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
            _M, _labels, _vmin, _vmax,
            f"{cmp_unit}<br>({_f(_vmin)}–{_f(_vmax)})",
            f"{cmp_metric.value} — {cmp_family.value}" + (" · clustered" if _clustered else " · plain"),
            side_groups=_side, blocks=_blocks, height=620)
        log_heat_blosum = mo.md(
            f"**{cmp_metric.value}** over **{_n} proteins** in `{cmp_family.value}` · "
            f"range {_f(_vmin)}–{_f(_vmax)} · labels are protein accessions · "
            + ("clustered by duplicate group (side strips = clusters, boxes = blocks)"
               if _clustered else "plain catalog order"))
    _out
    return (log_heat_blosum,)


@app.cell
def cmp_detail(
    PairwiseAligner,
    cmp_family,
    cmp_labels,
    cmp_matrix,
    mo,
    np,
    ssap_seqs,
    substitution_matrices,
):

    def _pair_alignment(la, lb):
        sa = ssap_seqs.get(la.split("|")[1])
        sb = ssap_seqs.get(lb.split("|")[1])
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
            _i, _j = np.unravel_index(np.nanargmax(_M), _M.shape)
            _la, _lb = cmp_labels[_i], cmp_labels[_j]
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


@app.cell(hide_code=True)
def uniq_net_hdr(mo):
    mo.md(r"""
    **Similarity map (PCoA)**

    **PCoA** is another method for visualizing pairwise alignments by plotting each sequence in a 2D graph along principal coordinates that allow for visible clusters of similar sequences. This uses the pairwise matrix of 100 - %identity to calculate the proper coordinates. Colors represent identified clusters of recombinase sequences.
    """)
    return


@app.cell
def uniq_net_family(mo, unique_msa):

    _famopts = sorted(unique_msa.keys(), key=lambda f: -len(unique_msa[f][0]))
    net_family = mo.ui.dropdown(options=_famopts,
                                value=(_famopts[0] if _famopts else None),
                                label="Family map")
    net_family
    return (net_family,)


@app.cell
def uniq_net(
    alt,
    cov_slider,
    disp_name,
    dup_id_slider,
    mo,
    net_family,
    np,
    pdf,
    pl,
    result,
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
            nrows.append({"protein": p, "genome": g, "species": disp_name(g),
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
        uniq_pcoa = None
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
                          tooltip=["protein", "genome", "species", "cluster"]))
        uniq_pcoa = mo.ui.altair_chart(
            chart_selection=False, legend_selection=False,
            chart=alt.layer(*_L).properties(
                width=640, height=470,
                title=f"{net_family.value} — PCoA of {len(_nodes)} proteins "
                      f"(hulls = duplicates ≥{dup_id_slider.value}% id / ≥{cov_slider.value} cov)").interactive())
        _out = uniq_pcoa
    result(_out)
    return convex_hull, pcoa_2d


@app.cell
def unique_fam_chart(alt, mo, pdf, result, unique_df):

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
                  .properties(height=max(140, 34 * _uf["family"].n_unique()), width=440,
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


@app.cell
def results_dedup(
    log_align,
    log_heat_blosum,
    log_msa,
    mo,
    result,
    sum_unique,
    sum_unique_fam,
):
    result(mo.accordion({"Results": mo.vstack([log_msa, sum_unique, sum_unique_fam, log_align, log_heat_blosum])}), title="Deduplication & similarity")
    return


@app.cell(hide_code=True)
def seed_exo_pfam_md(mo):
    mo.md(rf"""
    #### **Seed Exonuclease and Nuclease Inhibitor from Pfam:**
    In addition to recombinases, exonucleases and host-nuclease inhibitors can be searched:

    | Seed | Accession | Family |
    |------|-----------|--------|
    | PF09588 | YqaJ | exonuclease |
    | PF06064 | Gam | host-nuclease inhibitor |
    """)
    return


@app.cell(hide_code=True)
def seed_exo_md(mo):
    mo.md(rf"""
    #### **Seed Exonucleases for PSI-BLAST:**
    From literature, the following exonucleases are commonly integrated into new recombineering systems:

    | Seed | Accession | Family |
    |------|-----------|--------|
    | λExo / Redα | P03697 | YqaJ / λ-exonuclease (SynExo) |
    | RecE (nuclease domain) | P15032 | RecE toroidal exonuclease |
    """)
    return


@app.cell
def hmm_exo_hdr(mo):
    mo.md(r"""
    #### **3b. HMMER — exonuclease & host-nuclease inhibitor profiles**

    The same per-family HMMER search as 3a, filtered to the exonuclease / inhibitor
    Pfam profiles **YqaJ** (PF09588) and **Gam** (PF06064). Each profile keeps its own
    i-E-value cutoff, set alongside the recombinase profiles in 3a.
    """)

    return


@app.cell
def hmm_exo_config(
    ACCESSORY_FAMILIES,
    EXO_FAMILIES,
    hmm_evalue_exps,
    hmm_family_cutoffs,
    mo,
):
    _exo_fams = EXO_FAMILIES | ACCESSORY_FAMILIES
    _cutoffs_exo = hmm_family_cutoffs(hmm_evalue_exps)
    mo.md(
        "Per-family cutoffs are configured with the search in **3a** — "
        + ", ".join(f"**{fam}** i-E ≤ {_cutoffs_exo[fam]:.0e}" for fam in sorted(_exo_fams))
        + ". Change them there; the exonuclease / inhibitor tables below follow.")

    return


@app.cell(hide_code=True)
def hmm_exo_filter(
    ACCESSORY_FAMILIES,
    EXO_FAMILIES,
    hmm_evalue_exps,
    hmm_family_cutoffs,
    hmm_hits_all,
    mo,
    pl,
):
    _exo_fams = EXO_FAMILIES | ACCESSORY_FAMILIES
    _cutoffs_exo = hmm_family_cutoffs(hmm_evalue_exps)
    if hmm_hits_all.height:
        hmm_exo_hits_df = (
            hmm_hits_all.filter(pl.col("family").is_in(list(_exo_fams)))
            .with_columns(
                pl.col("family").replace_strict(_cutoffs_exo, default=0.0).alias("_i_e_cutoff"))
            .filter(pl.col("i_evalue") <= pl.col("_i_e_cutoff"))
            .sort("i_evalue")
            .unique(subset=["genome", "protein", "family"], keep="first")
            .sort(["family", "i_evalue"])
        )
    else:
        hmm_exo_hits_df = hmm_hits_all
    log_hmm_exo = mo.md(
        f"**{hmm_exo_hits_df.height} called exonuclease / inhibitor profile hits** at "
        + ", ".join(f"{fam} i-E ≤ {_cutoffs_exo[fam]:.0e}" for fam in sorted(_exo_fams))
        + f", in {hmm_exo_hits_df['genome'].n_unique() if hmm_exo_hits_df.height else 0} genomes.")

    return hmm_exo_hits_df, log_hmm_exo


@app.cell
def hmm_exo_table(
    ACCESSORY_FAMILIES,
    EXO_FAMILIES,
    hmm_evalue_exps,
    hmm_exo_hits_df,
    hmm_family_cutoffs,
    meta_df,
    mo,
    pl,
):
    _exo_fams = EXO_FAMILIES | ACCESSORY_FAMILIES
    _cutoffs_exo = hmm_family_cutoffs(hmm_evalue_exps)
    _hmm_exo = (hmm_exo_hits_df
                .join(meta_df.select([pl.col("accession").alias("genome"), "species"]),
                      on="genome", how="left")
                .select(["species", "genome", "protein", "family", "i_evalue", "dom_score",
                         "ali_from", "ali_to", "desc"])
                .sort("i_evalue"))
    sum_hmm_exo = mo.md(
        f"**{_hmm_exo.height} exonuclease / inhibitor hits** from HMMER at per-family "
        f"i-E cutoffs ("
        + ", ".join(f"{fam} ≤ {_cutoffs_exo[fam]:.0e}" for fam in sorted(_exo_fams))
        + f") · {_hmm_exo['genome'].n_unique() if _hmm_exo.height else 0} genomes.")
    mo.ui.table(_hmm_exo, selection=None, pagination=True, page_size=15)

    return (sum_hmm_exo,)


@app.cell(hide_code=True)
def psi_exo_hdr(mo):
    mo.md(r"""
    #### 3c. PSI-BLAST — exonuclease seeds

    A **separate** PSI-BLAST search using curated **exonuclease** seeds, run and
    tracked independently of the SSAP search, then merged by *role* into the
    catalog. The YqaJ HMM (PF09588 = λExo/YqaJ family) already catches the main
    exonuclease family; these seeds add sensitivity for divergent members and the
    RecE branch.

    | Seed | Accession | Family |
    |------|-----------|--------|
    | λExo / Redα | P03697 | YqaJ / λ-exonuclease (SynExo) |
    | RecE (nuclease domain) | P15032 | RecE toroidal exonuclease |
    """)
    return


@app.cell
def psi_exo_config(QUERY_DIR, RESULTS_DIR, mo):

    EXO_SEED_FAA = QUERY_DIR / "exo_seeds.faa"
    PSIBLAST_EXO_TSV = RESULTS_DIR / "psiblast_exo.tsv"

    psi_exo_evalue_exp = mo.ui.slider(start=-50, stop=1, step=1, value=-3,
                                      label="Keep hits with E ≤ 10^x", show_value=True)
    psi_exo_run_button = mo.ui.run_button(label="⟳ Re-run PSI-BLAST (exonuclease seeds)")

    mo.vstack([
        mo.hstack([psi_exo_evalue_exp, psi_exo_run_button], justify="start", gap=2),
        mo.md(f"seeds: `{EXO_SEED_FAA.name}` {'✅' if EXO_SEED_FAA.exists() else '⚠️'} · "
              f"cached: `{PSIBLAST_EXO_TSV.name}` {'✅' if PSIBLAST_EXO_TSV.exists() else '⚠️'}"),
    ])
    return (
        EXO_SEED_FAA,
        PSIBLAST_EXO_TSV,
        psi_exo_evalue_exp,
        psi_exo_run_button,
    )


@app.cell
def psi_exo_load(
    BLAST_DB,
    EXO_SEED_FAA,
    PSIBLAST_EXO_TSV,
    PSI_EVALUE,
    PSI_ITERATIONS,
    genome_set,
    mo,
    parse_psiblast_tsv,
    pl,
    psi_exo_run_button,
    subprocess,
):

    def _load_exo(do_rerun):
        log = []
        if do_rerun and EXO_SEED_FAA.exists():
            r = subprocess.run(
                ["psiblast", "-query", str(EXO_SEED_FAA), "-db", str(BLAST_DB),
                 "-num_iterations", str(PSI_ITERATIONS), "-evalue", str(PSI_EVALUE),
                 "-inclusion_ethresh", str(PSI_EVALUE), "-outfmt", "6 std qcovhsp",
                 "-num_threads", "4", "-out", str(PSIBLAST_EXO_TSV)],
                capture_output=True, text=True)
            log.append(f"psiblast rc={r.returncode}")
        schema = {"seed": pl.Utf8, "genome": pl.Utf8, "protein": pl.Utf8,
                  "pident": pl.Float64, "aln_len": pl.Int64, "evalue": pl.Float64,
                  "bitscore": pl.Float64, "qcovhsp": pl.Int64}
        rows = parse_psiblast_tsv(PSIBLAST_EXO_TSV) if PSIBLAST_EXO_TSV.exists() else []
        df = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
        if df.height:
            df = df.filter(pl.col("genome").is_in(list(genome_set)))
        return df, log

    psi_exo_hits_all, _exo_log = _load_exo(psi_exo_run_button.value)
    log_exo_load = mo.md(f"Parsed **{psi_exo_hits_all.height} exonuclease PSI-BLAST HSP rows**"
                         + (f"  \nRe-run: {', '.join(_exo_log)}" if _exo_log else ""))
    return log_exo_load, psi_exo_hits_all


@app.cell
def psi_exo_filter(meta_df, mo, pl, psi_exo_evalue_exp, psi_exo_hits_all):

    _thr = 10.0 ** psi_exo_evalue_exp.value
    if psi_exo_hits_all.height:
        psi_exo_hits_df = (
            psi_exo_hits_all.filter(pl.col("evalue") <= _thr)
            .sort("evalue")
            .unique(subset=["seed", "genome", "protein"], keep="first")
            .sort(["seed", "evalue"])
        )
    else:
        psi_exo_hits_df = psi_exo_hits_all

    _psi_exo = (psi_exo_hits_df
                .join(meta_df.select([pl.col("accession").alias("genome"), "species"]),
                      on="genome", how="left")
                .select(["species", "genome", "protein", "seed", "pident", "aln_len",
                         "qcovhsp", "evalue", "bitscore"])
                .sort("evalue"))
    sum_psi_exo = mo.md(
        f"**{_psi_exo.height} exonuclease hits** from PSI-BLAST at E ≤ {_thr:.0e} · "
        f"{_psi_exo['protein'].n_unique() if _psi_exo.height else 0} unique proteins across "
        f"{_psi_exo['genome'].n_unique() if _psi_exo.height else 0} genomes.")
    mo.ui.table(_psi_exo, selection=None, pagination=True, page_size=15)
    return psi_exo_hits_df, sum_psi_exo


@app.cell
def exo_hits_hdr(mo):
    mo.md(r"""
    #### Exonuclease & host-nuclease inhibitor hits — HMMER, PSI-BLAST, and merged

    The exonuclease / inhibitor hits the same way: **HMMER** (YqaJ, Gam),
    **PSI-BLAST**, and a **merged** per-protein table.
    """)
    return


@app.cell
def merged_exo_table(
    ACCESSORY_FAMILIES,
    EXO_FAMILIES,
    combine_role_hits,
    hmm_exo_hits_df,
    mo,
    psi_exo_hits_df,
):

    exo_hits = combine_role_hits(hmm_exo_hits_df, psi_exo_hits_df,
                                 EXO_FAMILIES | ACCESSORY_FAMILIES, "EXO-PSI")
    sum_merged_exo = mo.md(
        f"**{exo_hits.height} exonuclease / host-nuclease inhibitor hits** merging HMMER "
        f"(YqaJ, Gam) and PSI-BLAST across "
        f"{exo_hits['genome'].n_unique() if exo_hits.height else 0} genomes.")
    mo.ui.table(exo_hits, selection=None, pagination=True, page_size=15)
    return exo_hits, sum_merged_exo


@app.cell
def exo_hits_charts(alt, exo_hits, mo, pdf, pl, result):
    if exo_hits.height:
        _method_order = ["HMM", "PSI-BLAST", "Both"]
        _colors = alt.Scale(domain=_method_order, range=["#4c78a8", "#f58518", "#54a24b"])
        _exo = exo_hits.with_columns(
            pl.when(pl.col("method").str.contains("HMM") & pl.col("method").str.contains("PSI"))
            .then(pl.lit("Both"))
            .when(pl.col("method").str.contains("HMM")).then(pl.lit("HMM"))
            .otherwise(pl.lit("PSI-BLAST")).alias("method_label"))

        _by_species = (_exo.group_by(["species", "method_label"])
                       .agg(pl.col("protein").n_unique().alias("unique")))
        _species_chart = (alt.Chart(pdf(_by_species))
                          .mark_bar()
                          .encode(
                              x=alt.X("unique:Q", title="unique hits"),
                              y=alt.Y("species:N", sort="-x", title=None),
                              color=alt.Color("method_label:N", title="method", scale=_colors),
                              tooltip=["species", "method_label", "unique"])
                          .properties(height=max(180, 20 * _by_species["species"].n_unique()),
                                      width=400,
                                      title="Unique Exonuclease / Inhibitor Hits by Species"))

        _by_fam = (_exo.select(["protein", "method_label",
                                pl.col("hmm_families")
                                  .str.replace_all("—", "PSI-BLAST only (no Pfam)")
                                  .str.split(", ").alias("family")])
                   .explode("family", empty_as_null=True)
                   .group_by(["family", "method_label"])
                   .agg(pl.col("protein").n_unique().alias("unique")))
        _fam_chart = (alt.Chart(pdf(_by_fam))
                      .mark_bar()
                      .encode(
                          x=alt.X("unique:Q", title="unique hits"),
                          y=alt.Y("family:N", sort="-x", title=None),
                          color=alt.Color("method_label:N", title="method", scale=_colors),
                          tooltip=["family", "method_label", "unique"])
                      .properties(height=max(140, 30 * _by_fam["family"].n_unique()),
                                  width=400,
                                  title="Unique Exonuclease / Inhibitor Hits by Family"))
        _out = mo.hstack([mo.ui.altair_chart(_fam_chart), mo.ui.altair_chart(_species_chart)],
                         align="start", gap=2, widths="equal")
    else:
        _out = mo.md("*No merged exonuclease hits at the current thresholds.*")
    result(_out)

    return


@app.cell(hide_code=True)
def catalog_hdr(mo):
    mo.md(r"""
    #### **Recombinase catalog (SSAP ∪ EXO ∪ accessory)**

    The three searches are merged into one catalog, one row per (genome, protein):

    - **SSAP** = SSAP HMMs (RecT/ERF/Rad52/Sak) + SSAP PSI-BLAST (3a)
    - **EXO** = YqaJ HMM + exonuclease PSI-BLAST (3b)
    - **accessory** = Gam (PF06064), the host-nuclease inhibitor, tracked as a co-locator

    Each protein gets a **role** (SSAP / EXO / accessory / other) and a **method**
    tag (HMM / SSAP-PSI / EXO-PSI, combined with `+`). PSI-only SSAP hits are split
    by the seed that found them. The role-labelled catalog feeds comparison,
    uniqueness, and co-location.
    """)
    return


@app.cell
def recomb_catalog(
    ACCESSORY_FAMILIES,
    EXO_FAMILIES,
    SSAP_FAMILIES,
    hmm_hits_df,
    meta_df,
    mo,
    pl,
    psi_exo_hits_df,
    psi_hits_df,
    results_box,
):

    SEED_LABELS = {"P03698": "Red_beta", "P33228": "RecT_Ecoli", "Q7N2Y8": "Plu_beta",
                   "P03697": "lambda_exo", "P15032": "RecE"}

    def _agg_psi(df, seedcol, evcol):
        if df.height:
            return (df.group_by(["genome", "protein"]).agg(
                pl.col("seed").unique().sort().alias("_s"),
                pl.col("evalue").min().alias(evcol))
                .with_columns(pl.col("_s").list.join(",").alias(seedcol)).drop("_s"))
        return pl.DataFrame(schema={"genome": pl.Utf8, "protein": pl.Utf8,
                                    evcol: pl.Float64, seedcol: pl.Utf8})

    def _build_catalog(hmm_df, ssap_df, exo_df):
        hmm_side = (hmm_df.select(["genome", "protein", "family", "i_evalue", "desc"])
                    .rename({"i_evalue": "hmm_evalue"})) if hmm_df.height else pl.DataFrame(
            schema={"genome": pl.Utf8, "protein": pl.Utf8, "family": pl.Utf8,
                    "hmm_evalue": pl.Float64, "desc": pl.Utf8})
        cat = (hmm_side
               .join(_agg_psi(ssap_df, "ssap_seeds", "ssap_evalue"),
                     on=["genome", "protein"], how="full", coalesce=True)
               .join(_agg_psi(exo_df, "exo_seeds", "exo_evalue"),
                     on=["genome", "protein"], how="full", coalesce=True))
        cat = cat.with_columns(
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

        def _family(r):
            if r["family"]:
                return r["family"]
            if r["exo_seeds"]:
                return "YqaJ(PSI-exo)"
            if r["ssap_seeds"]:
                s = r["ssap_seeds"].split(",")[0]
                return f"SSAP(PSI:{SEED_LABELS.get(s, s)})"
            return "other(PSI)"

        cat = cat.with_columns(
            pl.Series("family", [_family(r) for r in cat.iter_rows(named=True)], dtype=pl.Utf8))
        return cat.sort(["role", "family", "hmm_evalue"])

    recomb_df = _build_catalog(hmm_hits_df, psi_hits_df, psi_exo_hits_df)

    _disp = recomb_df.join(
        meta_df.select([pl.col("accession").alias("genome"), "species", "strain"]),
        on="genome", how="left")
    _roles = ({r: int(n) for r, n in zip(
        recomb_df["role"].value_counts().sort("role")["role"],
        recomb_df["role"].value_counts().sort("role")["count"])}
        if recomb_df.height else {})
    mo.vstack([
        mo.ui.table(
            _disp.select(["role", "family", "method", "species", "genome", "protein",
                          "hmm_evalue", "ssap_evalue", "exo_evalue"]),
            selection=None, pagination=True, page_size=12),
        results_box(mo.md(
            f"**{recomb_df.height} recombineering proteins** across "
            f"{recomb_df['genome'].n_unique() if recomb_df.height else 0} genomes · roles: {_roles}")),
    ])
    return (recomb_df,)


@app.cell
def recomb_seqs(PROTEIN_FAA, load_protein_seqs, mo, recomb_df):
    recomb_seqs = load_protein_seqs(PROTEIN_FAA, set(recomb_df["protein"].to_list()))
    log_seqs = (mo.md(f"Loaded **{len(recomb_seqs)} / {recomb_df['protein'].n_unique()}** "
          f"recombinase protein sequences from `{PROTEIN_FAA.name}`."))

    return log_seqs, recomb_seqs


@app.cell
def results_searches(
    log_exo_load,
    log_hmm,
    log_hmm_exo,
    log_hmm_load,
    log_psi_load,
    log_seqs,
    mo,
    result,
    sum_hmm_exo,
    sum_hmm_rec,
    sum_merged_exo,
    sum_merged_rec,
    sum_psi_exo,
    sum_psi_rec,
    sum_superfam,
):
    result(mo.accordion({"Results": mo.vstack([log_hmm_load, log_hmm, log_hmm_exo, sum_hmm_rec, sum_psi_rec, sum_merged_rec, sum_superfam, log_psi_load, log_exo_load, sum_hmm_exo, sum_psi_exo, sum_merged_exo, log_seqs])}), title="Searches & catalog")
    return


@app.cell
def results_data_import(
    log_acq,
    log_blast,
    log_input,
    log_meta,
    log_proteins,
    log_refs,
    mo,
    result,
):
    result(mo.accordion({"Results": mo.vstack([log_input, log_acq, log_meta, log_proteins, log_blast, log_refs])}), title="Data import & database build")
    return


@app.cell(hide_code=True)
def step5_md(mo):
    mo.md(r"""
    ### 5. Identify putative operons containing combinations of recombinases and exonucleases
    With the unfiltered list of recombinases, exonucleases, and host-nuclease inhibitors, we can identify the location of the coding sequences (CDS) of each protein sequence in each genome. Then, each genome can be iteratively searched for co-located recombinase-exonuclease operons within a specified gap. These can then be visualized with a DNA feature viewer, showing the locations of the recombinase, exonuclease, other annotated proteins, and transcription start sites (TSS).
    """)
    return


@app.cell(hide_code=True)
def coloc_hdr(mo):
    mo.md(r"""
    #### 5a. Co-location & operons

    Recombineering systems are often an **SSAP + exonuclease** operon; some also
    carry the host-nuclease inhibitor **Gam**. We map each hit to its CDS
    coordinates in `genomes/<genome>/genomic.gff`, then scan **all genomes** for
    partners on the same contig within an adjustable **gap** (default 5 kb),
    recording the intergenic gap and orientation (SAME strand = operon-like;
    DIVERGENT = head-to-head). **Overlapping** CDSs are labelled `overlap` rather
    than clamped to zero. The optional unique filter keeps only pairs whose SSAP is
    a recombinase representative. GenBank-only drafts without a GFF (e.g. pyrodigal
    `_PRED` proteins) are reported as *unplaced*.
    """)
    return


@app.cell
def recomb_loci(GENOMES_DIR, collections, mo, pl, re, recomb_df):

    def _load_loci():
        rows, unplaced = [], []
        want = collections.defaultdict(dict)
        for r in recomb_df.iter_rows(named=True):
            want[r["genome"]][r["protein"]] = (r["family"], r["role"])
        for genome, wants in want.items():
            gff = GENOMES_DIR / genome / "genomic.gff"
            if not gff.exists():
                for p, (fam, role) in wants.items():
                    unplaced.append({"genome": genome, "protein": p, "family": fam, "role": role})
                continue
            found = set()
            with open(gff) as fh:
                for line in fh:
                    if line.startswith("#") or "\tCDS\t" not in line:
                        continue
                    c = line.rstrip("\n").split("\t")
                    m = re.search(r"protein_id=([^;]+)", c[8])
                    if not m or m.group(1) not in wants:
                        continue
                    pid = m.group(1)
                    lt = re.search(r"locus_tag=([^;]+)", c[8])
                    pr = re.search(r"product=([^;]+)", c[8])
                    fam, role = wants[pid]
                    rows.append({
                        "genome": genome, "protein": pid, "contig": c[0],
                        "start": int(c[3]), "end": int(c[4]), "strand": c[6],
                        "locus_tag": lt.group(1) if lt else "-",
                        "product": pr.group(1) if pr else "-",
                        "family": fam, "role": role,
                    })
                    found.add(pid)
            for p in set(wants) - found:
                unplaced.append({"genome": genome, "protein": p,
                                 "family": wants[p][0], "role": wants[p][1]})
        lschema = {"genome": pl.Utf8, "protein": pl.Utf8, "contig": pl.Utf8,
                   "start": pl.Int64, "end": pl.Int64, "strand": pl.Utf8,
                   "locus_tag": pl.Utf8, "product": pl.Utf8, "family": pl.Utf8, "role": pl.Utf8}
        loci = pl.DataFrame(rows, schema=lschema) if rows else pl.DataFrame(schema=lschema)
        return loci, unplaced

    recomb_loci_df, recomb_unplaced = _load_loci()
    log_loci = (mo.md(f"Placed **{recomb_loci_df.height}** recombineering proteins on GFF coordinates · "
          f"**{len(recomb_unplaced)}** unplaced (no GFF; pyrodigal drafts)."))
    return log_loci, recomb_loci_df


@app.cell
def coloc_controls(mo):

    gap_slider = mo.ui.slider(0, 20000, step=250, value=5000,
                              label="Max SSAP↔partner gap (bp)", show_value=True)
    unique_only_toggle = mo.ui.switch(value=False, label="Unique operons only (drop strain dupes)")
    mo.hstack([gap_slider, unique_only_toggle], justify="start", gap=2)
    return gap_slider, unique_only_toggle


@app.cell
def coloc_compute(
    collections,
    gap_slider,
    meta_df,
    mo,
    pl,
    recomb_loci_df,
    unique_only_toggle,
    unique_rep_set,
):

    def _find_operons(gap_max):
        by_genome = collections.defaultdict(list)
        for r in recomb_loci_df.iter_rows(named=True):
            by_genome[r["genome"]].append(r)
        rows = []
        for genome, genes in by_genome.items():
            ssaps = [g for g in genes if g["role"] == "SSAP"]
            partners = [g for g in genes if g["role"] in ("EXO", "accessory")]
            for s in ssaps:
                for e in partners:
                    if s["contig"] != e["contig"]:
                        continue
                    lo, hi = (s, e) if s["start"] <= e["start"] else (e, s)
                    gap = hi["start"] - lo["end"]
                    if gap > gap_max:
                        continue
                    rows.append({
                        "genome": genome, "contig": s["contig"],
                        "ssap_family": s["family"], "ssap_protein": s["protein"],
                        "partner_role": e["role"], "partner_family": e["family"],
                        "partner_protein": e["protein"],
                        "gap": gap, "overlap": gap < 0,
                        "orientation": "SAME" if s["strand"] == e["strand"] else "DIVERGENT",
                        "ssap_locus": s["locus_tag"], "partner_locus": e["locus_tag"],
                        "ssap_start": s["start"], "ssap_end": s["end"], "ssap_strand": s["strand"],
                        "partner_start": e["start"], "partner_end": e["end"],
                        "partner_strand": e["strand"],
                        "ssap_unique": f"{genome}|{s['protein']}" in unique_rep_set,
                    })
        oschema = ["genome", "contig", "ssap_family", "ssap_protein", "partner_role",
                   "partner_family", "partner_protein", "gap", "overlap", "orientation",
                   "ssap_locus", "partner_locus", "ssap_start", "ssap_end", "ssap_strand",
                   "partner_start", "partner_end", "partner_strand", "ssap_unique"]
        df = pl.DataFrame(rows, schema=oschema) if rows else pl.DataFrame(schema=oschema)
        return df.sort(["genome", "gap"]) if df.height else df

    operons_all = _find_operons(gap_slider.value)
    operons_df = (operons_all.filter(pl.col("ssap_unique"))
                  if (unique_only_toggle.value and operons_all.height) else operons_all)

    if operons_df.height:
        operon_summary = (operons_df.group_by(["genome", "ssap_protein", "ssap_family"])
                          .agg(pl.col("partner_role").unique().sort().alias("partners"),
                               pl.col("gap").min().alias("min_gap"))
                          .with_columns(
                              pl.col("partners").list.contains("EXO").alias("has_EXO"),
                              pl.col("partners").list.contains("accessory").alias("has_Gam"))
                          .with_columns(pl.col("partners").list.join("+").alias("partners"))
                          .sort(["genome", "min_gap"]))
    else:
        operon_summary = pl.DataFrame(schema={"genome": pl.Utf8, "ssap_protein": pl.Utf8,
                                              "ssap_family": pl.Utf8, "partners": pl.Utf8,
                                              "min_gap": pl.Int64, "has_EXO": pl.Boolean,
                                              "has_Gam": pl.Boolean})

    _ov = operons_df.filter(pl.col("overlap")).height if operons_df.height else 0
    sum_operons = mo.md(
        f"**{operons_df.height} SSAP↔partner pair(s)** within {gap_slider.value:,} bp "
        f"across {operons_df['genome'].n_unique() if operons_df.height else 0} genomes"
        + (" (unique only)" if unique_only_toggle.value else "")
        + f" · {operons_df.filter(pl.col('orientation') == 'SAME').height if operons_df.height else 0} SAME-strand"
        + f" · {_ov} overlapping")
    mo.ui.tabs({
        "SSAP–partner pairs": mo.ui.table(
            operons_df.join(meta_df.select([pl.col('accession').alias('genome'), 'species']),
                            on='genome', how='left')
            .select(['species', 'genome', 'ssap_family', 'ssap_protein', 'partner_role',
                     'partner_family', 'partner_protein', 'gap', 'orientation', 'ssap_unique'])
            if operons_df.height else operons_df,
            selection=None, pagination=True, page_size=15),
        "Per-SSAP summary": mo.ui.table(
            operon_summary.join(meta_df.select([pl.col('accession').alias('genome'), 'species']),
                                on='genome', how='left')
            if operon_summary.height else operon_summary,
            selection=None, pagination=True, page_size=15),
    })
    return operons_df, sum_operons


@app.cell(hide_code=True)
def view_hdr(mo):
    mo.md(r"""
    #### 5b. Sequence & locus viewer

    Inspect an operon's genomic neighbourhood as a gene-arrow map (all CDS from the
    GFF; **SSAP = blue, EXO = red, Gam = green, other = orange, context = grey**),
    plus the SSAP/EXO **protein domain maps** (HMM alignment envelope) and
    sequences. Annotated **regulatory features** (promoters/terminators/riboswitches)
    are drawn where a GFF carries them, with a note when none are present. Use the
    operon selector to step through every pair; an NCBI link jumps to the assembly.
    """)
    return


@app.cell
def viewer_utils(
    ACCESSORY_FAMILIES,
    EXO_FAMILIES,
    GENOMES_DIR,
    GraphicFeature,
    GraphicRecord,
    SSAP_FAMILIES,
    hmm_hits_all,
    mo,
    pl,
    plt,
    re,
    recomb_loci_df,
    recomb_seqs,
):

    ROLE_COLORS = {"SSAP": "#4c78a8", "EXO": "#e45756", "accessory": "#54a24b", "other": "#f58518"}
    REG_COLOR = "#9467bd"
    recomb_locus_map = {(r["genome"], r["protein"]): (r["family"], r["role"])
                        for r in recomb_loci_df.iter_rows(named=True)}

    def _gff_lines(genome):
        gff = GENOMES_DIR / genome / "genomic.gff"
        if not gff.exists():
            return
        with open(gff) as fh:
            for line in fh:
                if not line.startswith("#"):
                    yield line.rstrip("\n").split("\t")

    def viewer_cds_in_region(genome, contig, lo, hi):
        feats = []
        for c in _gff_lines(genome):
            if len(c) < 9 or c[2] != "CDS" or c[0] != contig:
                continue
            s, e = int(c[3]), int(c[4])
            if e < lo or s > hi:
                continue
            pid = re.search(r"protein_id=([^;]+)", c[8])
            pr = re.search(r"product=([^;]+)", c[8])
            feats.append({"start": s, "end": e, "strand": c[6],
                          "protein": pid.group(1) if pid else None,
                          "product": pr.group(1) if pr else ""})
        return feats

    def viewer_reg_features(genome, contig, lo, hi):
        """Annotated regulatory features (promoter/terminator/riboswitch/...) in a window."""
        feats = []
        for c in _gff_lines(genome):
            if len(c) < 9 or c[0] != contig:
                continue
            if c[2] not in ("regulatory_region", "regulatory", "promoter", "terminator", "TSS"):
                continue
            s, e = int(c[3]), int(c[4])
            if e < lo or s > hi:
                continue
            cls = re.search(r"regulatory_class=([^;]+)", c[8])
            note = re.search(r"(?:Note|product)=([^;]+)", c[8])
            label = cls.group(1) if cls else (note.group(1) if note else c[2])
            feats.append({"start": s, "end": e, "strand": c[6], "label": label})
        return feats

    def viewer_render_region(genome, contig, lo, hi, title):
        lo = max(1, lo)
        gfeats = []
        for f in viewer_cds_in_region(genome, contig, lo, hi):
            key = (genome, f["protein"])
            if key in recomb_locus_map:
                fam, role = recomb_locus_map[key]
                color, label = ROLE_COLORS.get(role, "#f58518"), f"{fam}\n{f['protein']}"
            else:
                color, label = "#c9c9c9", None
            gfeats.append(GraphicFeature(
                start=f["start"], end=f["end"],
                strand=(1 if f["strand"] == "+" else -1), color=color, label=label))
        for f in viewer_reg_features(genome, contig, lo, hi):
            gfeats.append(GraphicFeature(
                start=f["start"], end=f["end"], strand=0, color=REG_COLOR,
                label=f["label"]))
        rec = GraphicRecord(sequence_length=(hi - lo + 1), first_index=lo, features=gfeats)
        fig, ax = plt.subplots(figsize=(13, 3.2))
        rec.plot(ax=ax, with_ruler=True)
        ax.set_title(title, fontsize=10)
        return fig

    def viewer_domain_fig(genome, protein):
        doms = hmm_hits_all.filter((pl.col("genome") == genome) & (pl.col("protein") == protein))
        if not doms.height:
            return None
        tlen = int(doms["tlen"][0])
        feats = []
        for d in doms.iter_rows(named=True):
            col = ("#4c78a8" if d["family"] in SSAP_FAMILIES else
                   "#e45756" if d["family"] in EXO_FAMILIES else
                   "#54a24b" if d["family"] in ACCESSORY_FAMILIES else "#f58518")
            feats.append(GraphicFeature(
                start=d["ali_from"], end=d["ali_to"], strand=0, color=col,
                label=f"{d['family']} {d['ali_from']}-{d['ali_to']} (E={d['i_evalue']:.0e})"))
        rec = GraphicRecord(sequence_length=tlen, features=feats)
        fig, ax = plt.subplots(figsize=(11, 1.5))
        rec.plot(ax=ax, with_ruler=True)
        ax.set_title(f"{protein} — {tlen} aa", fontsize=9)
        return fig

    def viewer_seq_block(protein):
        s = recomb_seqs.get(protein)
        if not s:
            return mo.md(f"*sequence for {protein} not loaded*")
        wrapped = "\n".join(s[i:i + 60] for i in range(0, len(s), 60))
        return mo.md(f"**{protein}** ({len(s)} aa)\n```\n{wrapped}\n```")


    return (
        viewer_domain_fig,
        viewer_reg_features,
        viewer_render_region,
        viewer_seq_block,
    )


@app.cell
def view_controls(disp_name, mo, operons_df):

    def _operon_options(df):
        opts = {}
        for i, r in enumerate(df.iter_rows(named=True)):
            lab = (f"{disp_name(r['genome'])} [{r['genome']}] · {r['ssap_family']} {r['ssap_protein']}"
                   f" + {r['partner_family']} {r['partner_protein']} ({r['partner_role']})"
                   f"  (gap {r['gap']}bp, {r['orientation']})")
            opts[lab] = i
        return opts

    _opts = _operon_options(operons_df) if operons_df.height else {}
    operon_pick = mo.ui.dropdown(options=_opts, value=(next(iter(_opts)) if _opts else None),
                                 label="Operon to inspect", searchable=True)
    flank_slider = mo.ui.slider(1000, 25000, step=1000, value=8000,
                                label="Flank window (bp)", show_value=True)
    mo.hstack([operon_pick, flank_slider], justify="start", gap=2)
    return flank_slider, operon_pick


@app.cell
def view_map(
    flank_slider,
    mo,
    operon_pick,
    operons_df,
    result,
    viewer_reg_features,
    viewer_render_region,
):

    def _render_operon(op, flank):
        lo = max(1, min(op["ssap_start"], op["partner_start"]) - flank)
        hi = max(op["ssap_end"], op["partner_end"]) + flank
        return viewer_render_region(
            op["genome"], op["contig"], lo, hi,
            f"{op['genome']}  {op['contig']}:{lo:,}-{hi:,} "
            f"({op['ssap_family']}+{op['partner_family']}, {op['orientation']})")

    if operon_pick.value is None:
        view_map_out = mo.md("*No operons to display at the current gap threshold.*")
    else:
        _op = operons_df.row(operon_pick.value, named=True)
        _reg = viewer_reg_features(_op["genome"], _op["contig"],
                                   max(1, min(_op["ssap_start"], _op["partner_start"]) - flank_slider.value),
                                   max(_op["ssap_end"], _op["partner_end"]) + flank_slider.value)
        _link = f"https://www.ncbi.nlm.nih.gov/datasets/genome/{_op['genome']}/"
        view_map_out = mo.vstack([
            _render_operon(_op, flank_slider.value),
            mo.md(f"[↗ NCBI assembly record for {_op['genome']}]({_link})  ·  "
                  f"contig `{_op['contig']}`  ·  "
                  + (f"regulatory features in window: {', '.join(f['label'] for f in _reg)}"
                     if _reg else "*no annotated promoter/terminator features in this window*")),
        ])
    result(view_map_out)
    return


@app.cell
def view_domain(
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


@app.cell(hide_code=True)
def browse_hdr(mo):
    mo.md(r"""
    ### Browse any recombinase locus

    Pick a genome and one of its recombinases to view its neighbourhood arrow map,
    HMM domain map, and sequence — including recombinases that are **not** in an
    SSAP+EXO operon.
    """)
    return


@app.cell
def browse_controls(disp_label, mo, recomb_loci_df):
    _gopts = sorted(recomb_loci_df["genome"].unique().to_list())
    browse_genome = mo.ui.dropdown(
        options={disp_label(a): a for a in _gopts},
        value=(disp_label(_gopts[0]) if _gopts else None),
        label="Genome", searchable=True)
    browse_flank = mo.ui.slider(1000, 25000, step=1000, value=8000,
                                label="Flank (bp)", show_value=True)
    mo.hstack([browse_genome, browse_flank], justify="start", gap=2)
    return browse_flank, browse_genome


@app.cell
def browse_protein_ui(browse_genome, mo, pl, recomb_loci_df):
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
def browse_view(
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
            f"{_row['genome']}  {_row['contig']}:{_lo:,}-{_hi:,}  "
            f"[{_row['family']} {_row['protein']}]")
        _dfig = viewer_domain_fig(_row["genome"], _row["protein"])
        _link = f"https://www.ncbi.nlm.nih.gov/datasets/genome/{_row['genome']}/"
        _parts = [_fig]
        if _dfig is not None:
            _parts += [mo.md("**Domain map**"), _dfig]
        _parts += [viewer_seq_block(_row["protein"]),
                   mo.md(f"[↗ NCBI assembly record]({_link}) · locus_tag "
                         f"`{_row['locus_tag']}` · product: {_row['product']}")]
        browse_out = mo.vstack(_parts)
    result(browse_out)
    return


@app.cell
def results_operons(log_loci, mo, result, sum_operons):
    result(mo.accordion({"Results": mo.vstack([log_loci, sum_operons])}), title="Co-location & operons")
    return


@app.cell(hide_code=True)
def step6_md(mo):
    mo.md(r"""
    ### 6. Evaluate recombinases and exonucleases based off of their genomes relation to _X. autotrophicus_ GJ10
    Finally, after identifying a list of recombinases and recombinase-exonuclease systems, we can further rank their importance based off of their respective genomes. Specifically, similar to how recombinases can be analyzed through % similarity, genomes can be analyzed using **Average Nucleotide Identity (ANI)** and **Alignment Fraction (AF)**, which creates a pairwise matrix comparing the sequence identity of each genome. This can be calculated efficiently using either **skani** or **FastANI**.

    These pairwise matrices can be visualized with heatmaps and PCoA in order to manually adjust thresholds for duplicates (99.5%), species (95%), and evaluate relationships between different _Xanthobacter_ species.

    Following deduplication, each genome within a group can be weighted based off of a variety of parameters quantifying the quality of the genome, relationship to a specific strain (_X. autotrophicus_ GJ10), and the presence of recombineering systems. This results in a final table ranking each genome and respective recombinase system, allowing the user to select their top options for recombineering systems.
    """)
    return


@app.cell(hide_code=True)
def ani_hdr(mo):
    mo.md(r"""
    #### 6a. ANI explorer — how the genome set clusters

    Average Nucleotide Identity (ANI) and alignment fraction (AF) were computed
    with **skani** (per-genome, symmetrised by taking the *minimum* of the two
    directional values), and cross-checked against **fastANI** on the complete
    genomes. Here we load the precomputed symmetric matrices
    (`ani/ani_matrix.tsv`, `ani/af_matrix.tsv`) and explore how the genomes
    group, which assemblies are near-duplicates, and how representatives are
    chosen — all with live thresholds.

    - **Duplicate** ≈ same assembly: ANI ≥ *dup-ANI* **and** AF ≥ *dup-AF*
      (tiny assemblies are never merged).
    - **Species cluster**: single-linkage at ANI ≥ *species-threshold*
      (no AF requirement).
    - **Representative** of a group: prefer RefSeq (GCF) > GenBank (GCA), then
      Complete > draft, then fewer contigs, then longer sequence.
    """)
    return


@app.cell
def ani_controls(mo):

    ani_run_button = mo.ui.run_button(label="⟳ Recompute ANI with skani (all genomes)")
    ani_use_fastani = mo.ui.switch(value=False, label="Cross-check with fastANI (complete genomes)")
    mo.hstack([ani_run_button, ani_use_fastani], justify="start", gap=2)
    return ani_run_button, ani_use_fastani


@app.cell
def ani_recompute(
    ACC_RE,
    ANI_DIR,
    Path,
    ani_run_button,
    ani_use_fastani,
    meta_df,
    mo,
    np,
    pl,
    subprocess,
):

    _ani_log = []

    def _acc_of_path(p):
        m = ACC_RE.search(Path(p).name)
        return m.group(0) if m else Path(p).stem

    def _run_skani():
        fnas = [r["path"] for r in meta_df.iter_rows(named=True) if r["path"]]
        if len(fnas) < 2:
            return "skani skipped (need ≥2 genome FASTAs)"
        out = ANI_DIR / "skani.tsv"
        r = subprocess.run(["skani", "triangle", "-E", "--min-af", "0", "-o", str(out)] + fnas,
                           capture_output=True, text=True)
        return (f"skani rc={r.returncode} over {len(fnas)} genomes"
                + ("" if r.returncode == 0 else f" — {(r.stderr or '')[-200:]}"))

    def _run_fastani():
        fnas = [r["path"] for r in meta_df.iter_rows(named=True) if r["path"]]
        if len(fnas) < 2:
            return "fastANI skipped"
        ql = ANI_DIR / "fastani_list.txt"
        ql.write_text("\n".join(fnas) + "\n")
        out = ANI_DIR / "fastani.tsv"
        r = subprocess.run(["fastANI", "--ql", str(ql), "--rl", str(ql), "-o", str(out)],
                           capture_output=True, text=True)
        return f"fastANI rc={r.returncode}"

    def _skani_to_matrices():
        raw = ANI_DIR / "skani.tsv"
        if not raw.exists():
            return None
        df = pl.read_csv(raw, separator="\t", infer_schema_length=0)
        accs = sorted({_acc_of_path(p) for p in df["Ref_file"].to_list()}
                      | {_acc_of_path(p) for p in df["Query_file"].to_list()})
        idx = {a: i for i, a in enumerate(accs)}
        n = len(accs)
        A = np.full((n, n), np.nan); F = np.full((n, n), np.nan)
        for r in df.iter_rows(named=True):
            i, j = idx[_acc_of_path(r["Ref_file"])], idx[_acc_of_path(r["Query_file"])]
            try:
                A[i, j] = float(r["ANI"])
                F[i, j] = float(r["Align_fraction_query"])
            except (TypeError, ValueError):
                continue
        np.fill_diagonal(A, 100.0); np.fill_diagonal(F, 100.0)
        A, F = np.fmin(A, A.T), np.fmin(F, F.T)

        def _write(path, M):
            with open(path, "w") as fh:
                fh.write("accession\t" + "\t".join(accs) + "\n")
                for a, row in zip(accs, M):
                    fh.write(a + "\t" + "\t".join("" if np.isnan(v) else f"{v:.2f}" for v in row) + "\n")
        _write(ANI_DIR / "ani_matrix.tsv", A)
        _write(ANI_DIR / "af_matrix.tsv", F)
        return len(accs)

    if ani_run_button.value:
        _ani_log.append(_run_skani())
        if ani_use_fastani.value:
            _ani_log.append(_run_fastani())

    if (ANI_DIR / "skani.tsv").exists() and (ani_run_button.value or not (ANI_DIR / "ani_matrix.tsv").exists()):
        _n = _skani_to_matrices()
        if _n:
            _ani_log.append(f"symmetric ANI/AF matrices for {_n} genomes")

    log_ani_recompute = (mo.vstack([
        mo.md("**ANI recompute** — " + ("; ".join(_ani_log) if _ani_log
              else "cached matrices in use (tick the button to recompute with skani)")),
    ]))
    return (log_ani_recompute,)


@app.cell
def ani_core(ANI_DIR, genome_set, leaves_list, linkage, mo, np, pl):

    GJ10_ACCS = ["GCF_020683165.1", "GCF_060823235.1"]

    def _load_sym_matrix(path, accs_keep):
        df = pl.read_csv(path, separator="\t", infer_schema_length=0)
        row_accs = df["accession"].to_list()
        col_accs = df.columns[1:]
        vals = df.select([pl.col(c).cast(pl.Float64, strict=False) for c in col_accs]).to_numpy()
        cpos = {a: i for i, a in enumerate(col_accs)}
        vals = vals[:, [cpos[a] for a in row_accs]]
        keep = [i for i, a in enumerate(row_accs) if a in accs_keep]
        accs = [row_accs[i] for i in keep]
        return accs, vals[np.ix_(keep, keep)]

    _ani_path = ANI_DIR / "ani_matrix.tsv"
    _af_path = ANI_DIR / "af_matrix.tsv"
    _have_ani = _ani_path.exists() and _af_path.exists()

    if _have_ani:
        accs_ani, ANI = _load_sym_matrix(_ani_path, genome_set)
        _accs_af, AF = _load_sym_matrix(_af_path, genome_set)
        assert _accs_af == accs_ani, "ANI/AF accession order mismatch"
        n_ani = len(accs_ani)
        ani_idx = {a: i for i, a in enumerate(accs_ani)}
        _D = np.where(np.isnan(ANI), 75.0, ANI)
        _dist = 1.0 - _D / 100.0
        np.fill_diagonal(_dist, 0.0)
        _dist = (_dist + _dist.T) / 2.0
        ani_linkage = linkage(_dist[np.triu_indices(n_ani, 1)], method="average")
        ani_order = list(leaves_list(ani_linkage))
    else:
        accs_ani, ANI, AF, n_ani, ani_idx, ani_linkage, ani_order = [], None, None, 0, {}, None, []

    log_ani = (mo.md(
        f"Loaded ANI/AF matrices for **{n_ani} genomes** "
        f"({', '.join(a for a in GJ10_ACCS if a in ani_idx) or 'no'} = *X. flavus* GJ10)."
        if _have_ani else
        "⚠️ `ani/ani_matrix.tsv` / `af_matrix.tsv` not found — run the ANI recompute above."
    ))
    return AF, ANI, GJ10_ACCS, accs_ani, ani_idx, ani_order, log_ani, n_ani


@app.cell
def genome_layout(ANI, accs_ani, mo, np, pcoa_2d, pl):

    def _genome_layout():
        D = 1.0 - ANI / 100.0
        _fin = D[np.isfinite(D)]
        _fill = (float(np.nanmax(D)) + 0.02) if _fin.size else 0.5
        Df = np.where(np.isfinite(D), D, _fill)
        np.fill_diagonal(Df, 0.0)
        Df = (Df + Df.T) / 2.0
        if Df.shape[0] == 0:
            return np.zeros((0, 2))
        return pcoa_2d(Df)

    _xy = _genome_layout()
    genome_xy_df = pl.DataFrame({"accession": accs_ani,
                                 "x": _xy[:, 0].tolist(), "y": _xy[:, 1].tolist()})
    log_pcoa = (mo.md(f"**PCoA** of the ANI distance for all **{len(accs_ani)} genomes** "
          f"(deterministic classical MDS; positions fixed, thresholds only recolour)."))
    return genome_xy_df, log_pcoa


@app.cell
def ani_controls(mo):

    dup_ani_slider = mo.ui.slider(start=95.0, stop=100.0, step=0.1, value=99.5,
                                  label="Duplicate ANI ≥ (%)", show_value=True)
    dup_af_slider = mo.ui.slider(start=50.0, stop=100.0, step=1.0, value=90.0,
                                 label="Duplicate AF ≥ (%)", show_value=True)
    species_slider = mo.ui.slider(start=80.0, stop=100.0, step=0.5, value=95.0,
                                  label="Species ANI ≥ (%)", show_value=True)
    ani_label_mode = mo.ui.radio(options=["accession", "species"], value="accession",
                                 label="Heatmap labels", inline=True)
    ani_heat_order = mo.ui.radio(
        options=["Clustered (ANI linkage)", "Plain (by species)"],
        value="Clustered (ANI linkage)", label="Heatmap ordering", inline=True)
    mo.vstack([mo.md("**Live thresholds & ordering** — drag to recompute clusters & representatives"),
               dup_ani_slider, dup_af_slider, species_slider, ani_label_mode, ani_heat_order])
    return (
        ani_heat_order,
        ani_label_mode,
        dup_af_slider,
        dup_ani_slider,
        species_slider,
    )


@app.cell
def ani_clusters(
    AF,
    ANI,
    accs_ani,
    collections,
    dup_af_slider,
    dup_ani_slider,
    meta_rows,
    n_ani,
    np,
    pl,
    species_slider,
):
    def _compute_clusters(dup_ani, dup_af, sp_thr):
        def _uf(k):
            return list(range(k))
        def _find(p, x):
            while p[x] != x:
                p[x] = p[p[x]]; x = p[x]
            return x
        def _union(p, a, b):
            ra, rb = _find(p, a), _find(p, b)
            if ra != rb:
                p[rb] = ra
        tiny = {i for i, a in enumerate(accs_ani) if meta_rows.get(a, {}).get("is_tiny")}

        def rep_key(i):
            a = accs_ani[i]; r = meta_rows.get(a, {})
            return (0 if a.startswith("GCF") else 1,
                    0 if r.get("level") == "Complete Genome" else 1,
                    r.get("contigs") or 10**9, -(r.get("length") or 0))

        def reason(i):
            a = accs_ani[i]; r = meta_rows.get(a, {})
            return ", ".join([
                "RefSeq" if a.startswith("GCF") else "GenBank",
                "Complete" if r.get("level") == "Complete Genome" else (r.get("level") or "draft"),
                f"{r.get('contigs', '?')} contigs",
                f"{(r.get('length') or 0):,} bp",
            ])

        # duplicate groups (tiny never merged)
        dpar = _uf(n_ani)
        for i in range(n_ani):
            for j in range(i + 1, n_ani):
                if i in tiny or j in tiny:
                    continue
                a, f = ANI[i, j], AF[i, j]
                if np.isnan(a) or a < dup_ani or np.isnan(f) or f < dup_af:
                    continue
                _union(dpar, i, j)
        groups = collections.defaultdict(list)
        for i in range(n_ani):
            groups[_find(dpar, i)].append(i)

        reps = set(); rows = []; gid = 0
        for _root, members in sorted(groups.items()):
            rep = min(members, key=rep_key)
            reps.add(accs_ani[rep])
            if len(members) < 2:
                continue
            gid += 1
            pa = [ANI[i, j] for i in members for j in members if i < j and not np.isnan(ANI[i, j])]
            pf = [AF[i, j] for i in members for j in members if i < j and not np.isnan(AF[i, j])]
            rows.append({
                "group": gid, "n": len(members),
                "members": ",".join(accs_ani[i] for i in sorted(members)),
                "representative": accs_ani[rep], "why_rep": reason(rep),
                "min_ANI": round(min(pa), 2) if pa else None,
                "min_AF": round(min(pf), 2) if pf else None,
                "has_tiny": any(i in tiny for i in members),
            })
        dup_df = (pl.DataFrame(rows) if rows else pl.DataFrame(
            schema={"group": pl.Int64, "n": pl.Int64, "members": pl.Utf8,
                    "representative": pl.Utf8, "why_rep": pl.Utf8}))

        # species clusters (single-linkage, no AF)
        spar = _uf(n_ani)
        for i in range(n_ani):
            for j in range(i + 1, n_ani):
                a = ANI[i, j]
                if not np.isnan(a) and a >= sp_thr:
                    _union(spar, i, j)
        sp = collections.defaultdict(list)
        for i in range(n_ani):
            sp[_find(spar, i)].append(i)
        sprows = []
        sp_of = {}
        for cid, (_r, members) in enumerate(sorted(sp.items(), key=lambda kv: -len(kv[1])), 1):
            rep = min(members, key=rep_key)
            for i in members:
                sp_of[i] = cid
            sprows.append({
                "cluster": cid, "n": len(members),
                "species": meta_rows.get(accs_ani[rep], {}).get("species", "?"),
                "representative": accs_ani[rep],
                "members": ",".join(accs_ani[i] for i in sorted(members)),
            })
        # per-genome membership (dup group id, representative, kept/merged, species cluster)
        grp_id_of, grp_rep_of, gid2 = {}, {}, 0
        for _root, members in sorted(groups.items()):
            rep = min(members, key=rep_key)
            if len(members) >= 2:
                gid2 += 1
                for i in members:
                    grp_id_of[i] = gid2; grp_rep_of[i] = rep
            else:
                grp_rep_of[members[0]] = members[0]
        memb = [{
            "accession": accs_ani[i],
            "dup_group": grp_id_of.get(i),
            "dup_rep": accs_ani[grp_rep_of.get(i, i)],
            "is_kept": accs_ani[i] in reps,
            "species_cluster": sp_of.get(i),
            "species": meta_rows.get(accs_ani[i], {}).get("species", "?"),
        } for i in range(n_ani)]
        membership = pl.DataFrame(memb)
        return dup_df, pl.DataFrame(sprows), reps, gid, rep_key, membership

    (dup_groups_df, species_clusters_df, ani_reps, n_dup_groups,
     ani_rep_key, ani_membership) = _compute_clusters(
        dup_ani_slider.value, dup_af_slider.value, species_slider.value)
    n_reps = len(ani_reps)
    return (
        ani_membership,
        ani_rep_key,
        dup_groups_df,
        n_dup_groups,
        n_reps,
        species_clusters_df,
    )


@app.cell
def ani_summary(
    dup_af_slider,
    dup_ani_slider,
    dup_groups_df,
    mo,
    n_ani,
    n_dup_groups,
    n_reps,
    species_clusters_df,
    species_slider,
):

    _msg = (
        f"**{n_ani} genomes → {n_reps} representatives** "
        f"({n_dup_groups} duplicate group(s) collapsed at ANI ≥ {dup_ani_slider.value:.1f}% "
        f"& AF ≥ {dup_af_slider.value:.0f}%) · "
        f"**{species_clusters_df.height} species clusters** at ANI ≥ {species_slider.value:.1f}%"
    )
    sum_ani = mo.md(_msg)
    mo.ui.tabs({
        "Duplicate groups (representative choice)":
            mo.ui.table(dup_groups_df, selection=None, pagination=True, page_size=10),
        "Species clusters":
            mo.ui.table(species_clusters_df, selection=None, pagination=True, page_size=10),
    })
    return (sum_ani,)


@app.cell
def ani_heatmap(
    ANI,
    GJ10_ACCS,
    accs_ani,
    ani_heat_order,
    ani_label_mode,
    ani_order,
    disp_name,
    matrix_heatmap_plotly,
    meta_rows,
    mo,
    n_ani,
    np,
):

    def _ani_parts(order_mode):
        if order_mode.startswith("Clustered"):
            _ord = list(ani_order); _note = "average-linkage order"
        else:
            _ord = sorted(range(n_ani),
                          key=lambda i: ((meta_rows.get(accs_ani[i], {}).get("species") or "?"),
                                         accs_ani[i]))
            _note = "ordered by species"
        return _ord, _note

    _ord, _note = _ani_parts(ani_heat_order.value)
    _labels = [accs_ani[i] for i in _ord]
    _M = ANI[np.ix_(_ord, _ord)].astype(float)
    _n = len(_labels)
    _vmin = float(np.nanmin(_M))
    _side = [(meta_rows.get(a, {}).get("species") or "?") for a in _labels]
    _tk = [disp_name(a) for a in _labels] if ani_label_mode.value == "species" else _labels
    _blocks = [(_i, _i, "#d62728", 1.8) for _i, _a in enumerate(_labels) if _a in GJ10_ACCS]

    fig = matrix_heatmap_plotly(
        _M, _tk, _vmin, 100.0, f"ANI (%)<br>({_vmin:.1f}–100)",
        f"Xanthobacter ANI — {_n} genomes ({_note})",
        side_groups=_side, blocks=_blocks, height=720, width=780, colorscale="YlOrRd")
    log_heat_ani = mo.md(
        f"ANI over **{_n} genomes** ({_note}) · scale {_vmin:.1f}–100% · species colour strips "
        f"top/left; red boxes mark ***X. flavus* GJ10** "
        f"(`{'`, `'.join(a for a in GJ10_ACCS if a in set(_labels))}`).")
    fig
    return (log_heat_ani,)


@app.cell(hide_code=True)
def ani_pcoa_hdr(mo):
    mo.md(r"""
    #### 6c. Genome map (PCoA) & relationships

    A deterministic **PCoA** of the ANI distance for **all** genomes. Positions are
    fixed; colour and hulls update live with the thresholds. Points are uniform
    circles (no representative/merged encoding); **hulls** group duplicate or
    species clusters (per the toggle). ***X. flavus* GJ10** is marked. Explore
    nearest neighbours and representative choices below.
    """)
    return


@app.cell
def ani_map_controls(mo):

    map_color_mode = mo.ui.radio(options=["duplicate groups", "species clusters"],
                                 value="duplicate groups", label="Colour nodes by", inline=True)
    show_hulls = mo.ui.switch(value=True, label="Draw hulls around groups")
    mo.hstack([map_color_mode, show_hulls], justify="start", gap=2)
    return map_color_mode, show_hulls


@app.cell
def ani_pcoa(
    GJ10_ACCS,
    alt,
    ani_membership,
    collections,
    convex_hull,
    genome_xy_df,
    map_color_mode,
    mo,
    pdf,
    pl,
    result,
    show_hulls,
):

    def _genome_map():
        nodes = (genome_xy_df.join(ani_membership, on="accession", how="inner")
                 .with_columns(
                     pl.col("species").str.replace("Xanthobacter", "X.").alias("species"),
                     pl.col("dup_group").cast(pl.Utf8).fill_null("— (unique)").alias("dup_group_lbl"),
                     pl.when(pl.col("accession").is_in(GJ10_ACCS)).then(pl.lit("GJ10"))
                       .otherwise(pl.lit("")).alias("gj10")))
        return nodes

    _nodes = _genome_map()
    if _nodes is None or not _nodes.height:
        ani_pcoa = None
        _out = mo.md("*No genome layout available.*")
    else:
        _cf = "dup_group_lbl" if map_color_mode.value == "duplicate groups" else "species"
        _hrows = []
        if show_hulls.value:
            _groups = collections.defaultdict(list)
            for r in _nodes.iter_rows(named=True):
                _groups[r[_cf]].append((r["x"], r["y"]))
            for g, pts in _groups.items():
                if len(pts) >= 3:
                    for h in convex_hull(pts):
                        _hrows.append({"x": h[0], "y": h[1], "group": g})
        _L = []
        if _hrows:
            _L.append(alt.Chart(pdf(pl.DataFrame(_hrows))).mark_line(opacity=0.2, strokeWidth=1)
                      .encode(x="x:Q", y="y:Q", detail="group:N",
                              color=alt.Color("group:N", legend=None)))
        _L.append(alt.Chart(pdf(_nodes)).mark_circle(size=110, filled=True, opacity=0.85)
                  .encode(x=alt.X("x:Q", title="PCoA 1"), y=alt.Y("y:Q", title="PCoA 2"),
                          color=alt.Color(f"{_cf}:N",
                                          legend=alt.Legend(columns=2, labelFontSize=8, symbolLimit=100)),
                          tooltip=["accession", "species", "dup_group_lbl", "species_cluster"]))
        _gj = _nodes.filter(pl.col("gj10") == "GJ10")
        if _gj.height:
            _L.append(alt.Chart(pdf(_gj)).mark_point(shape="diamond", size=260, filled=False,
                                                     stroke="#d62728", strokeWidth=2.5)
                      .encode(x="x:Q", y="y:Q", tooltip=["accession"]))
        ani_pcoa = mo.ui.altair_chart(
            chart_selection=False, legend_selection=False,
            chart=alt.layer(*_L).properties(
                width=660, height=500,
                title=f"Genome map (PCoA) — colour = {map_color_mode.value}; "
                      f"◆ red = X. flavus GJ10").interactive())
        _out = ani_pcoa
    result(_out)
    return


@app.cell(hide_code=True)
def ani_focus_hdr(mo):
    mo.md(r"""
    ### Explore relationships & representative choice

    Pick a focus genome for its nearest neighbours (ANI + AF), whether each would
    be **merged as a duplicate** at current thresholds, and whether it shares the
    focus genome's **species cluster**. Then open a duplicate group to see exactly
    *why* its representative was chosen (the tie-break ranking).
    """)
    return


@app.cell
def ani_focus_ui(accs_ani, disp_label, mo):
    nn_pick = mo.ui.dropdown(
        options={disp_label(a): a for a in accs_ani},
        value=disp_label(accs_ani[0]),
        label="Focus genome", searchable=True)
    nn_pick
    return (nn_pick,)


@app.cell
def ani_focus_table(
    AF,
    ANI,
    accs_ani,
    ani_idx,
    dup_af_slider,
    dup_ani_slider,
    meta_rows,
    mo,
    n_ani,
    nn_pick,
    np,
    pl,
    species_clusters_df,
):

    def _nn_table(qi):
        sc_members, sc_id = set(), None
        for r in species_clusters_df.iter_rows(named=True):
            ms = r["members"].split(",")
            if nn_pick.value in ms:
                sc_members, sc_id = set(ms), r["cluster"]; break
        da, afthr = dup_ani_slider.value, dup_af_slider.value
        tinyset = {a for a in accs_ani if meta_rows.get(a, {}).get("is_tiny")}
        focus_tiny = nn_pick.value in tinyset
        rows = []
        for j in range(n_ani):
            if j == qi:
                continue
            a = ANI[qi, j]
            if np.isnan(a):
                continue
            aj = accs_ani[j]; r = meta_rows.get(aj, {}); fq = AF[qi, j]
            rows.append({
                "accession": aj, "ANI": round(float(a), 2),
                "AF": None if np.isnan(fq) else round(float(fq), 1),
                "species": r.get("species"), "strain": r.get("strain"),
                "same_species_cluster": aj in sc_members,
                "would_merge_as_dup": (a >= da and not np.isnan(fq) and fq >= afthr
                                       and aj not in tinyset and not focus_tiny),
            })
        return pl.DataFrame(rows).sort("ANI", descending=True), sc_id, focus_tiny

    nn_df, _sc_id, _focus_tiny = _nn_table(ani_idx[nn_pick.value])
    _f = meta_rows.get(nn_pick.value, {})
    sum_nn = mo.md(
        f"**Nearest neighbours of `{nn_pick.value}`** — "
        f"{_f.get('species', '?')} {_f.get('strain', '') or ''} · "
        f"species cluster {_sc_id} · {'tiny (never merged)' if _focus_tiny else 'mergeable'}")
    mo.ui.table(nn_df, selection=None, pagination=True, page_size=12)
    return (sum_nn,)


@app.cell
def ani_dup_ui(dup_groups_df, mo):
    _gids = [str(g) for g in dup_groups_df["group"].to_list()] if dup_groups_df.height else []
    dup_pick = mo.ui.dropdown(options=_gids, value=(_gids[0] if _gids else None),
                              label="Duplicate group to inspect")
    dup_pick
    return (dup_pick,)


@app.cell
def ani_dup_detail(
    ani_idx,
    ani_rep_key,
    dup_groups_df,
    dup_pick,
    meta_rows,
    mo,
    pl,
):

    def _dup_detail(gid):
        row = dup_groups_df.filter(pl.col("group") == int(gid)).to_dicts()[0]
        members = row["members"].split(",")
        ranked = sorted(members, key=lambda a: ani_rep_key(ani_idx[a]))
        rows = []
        for rank, a in enumerate(ranked, 1):
            r = meta_rows.get(a, {})
            rows.append({
                "rank": rank, "accession": a, "chosen": "✓" if a == ranked[0] else "",
                "source": r.get("source"), "level": r.get("level"),
                "contigs": r.get("contigs"), "length": r.get("length"),
                "is_tiny": r.get("is_tiny"),
            })
        return pl.DataFrame(rows), row, ranked[0]

    if dup_pick.value is None:
        sum_dup = mo.md("*No duplicate groups at the current thresholds.*")
        _out = sum_dup
    else:
        _tbl, _row, _rep = _dup_detail(dup_pick.value)
        sum_dup = mo.md(
            f"**Group {_row['group']}** — {_row['n']} assemblies, "
            f"min ANI {_row['min_ANI']}%, min AF {_row['min_AF']}%.  \n"
            f"**Representative: `{_rep}`** ({_row['why_rep']}).  \n"
            f"Tie-break order: RefSeq (GCF) > GenBank (GCA) → Complete > draft → "
            f"fewer contigs → longer sequence. Ranked above (rank 1 = chosen).")
        _out = mo.vstack([mo.ui.table(_tbl, selection=None), sum_dup])
    _out
    return (sum_dup,)


@app.cell
def rank_controls(mo):

    _w_defaults = {"quality": 0.20, "ani": 0.30, "exo": 0.25, "gam": 0.05, "nrec": 0.20}

    w_quality = mo.ui.slider(0.0, 1.0, step=0.05, value=_w_defaults["quality"],
                             label="importance · genome quality")
    w_ani = mo.ui.slider(0.0, 1.0, step=0.05, value=_w_defaults["ani"],
                         label="importance · ANI to GJ10")
    w_exo = mo.ui.slider(0.0, 1.0, step=0.05, value=_w_defaults["exo"],
                         label="importance · has EXO partner")
    w_gam = mo.ui.slider(0.0, 1.0, step=0.05, value=_w_defaults["gam"],
                         label="importance · has Gam")
    w_nrec = mo.ui.slider(0.0, 1.0, step=0.05, value=_w_defaults["nrec"],
                          label="importance · # unique recombinases")

    mo.vstack([
        mo.hstack([w_quality, w_ani, w_exo, w_gam, w_nrec], justify="start", gap=2, wrap=True),
        mo.md("Sliders are **relative importance**: they are normalised downstream, so the "
              "score always uses weights that sum to exactly 1.000 (table in 6d)."),
    ])
    return w_ani, w_exo, w_gam, w_nrec, w_quality


@app.cell
def ranking(
    ANI,
    GJ10_ACCS,
    ani_idx,
    cov_slider,
    dup_id_slider,
    meta_df,
    mo,
    np,
    operons_df,
    pl,
    rep_of,
    single_linkage,
    unique_df,
    unique_msa,
    w_ani,
    w_exo,
    w_gam,
    w_nrec,
    w_quality,
):

    def _minmax(vals):
        v = [x for x in vals if x is not None and np.isfinite(x)]
        if not v:
            return lambda x: 0.0
        lo, hi = min(v), max(v)
        if hi <= lo:
            return lambda x: 1.0
        return lambda x: 0.0 if x is None or not np.isfinite(x) else (float(x) - lo) / (hi - lo)

    _LVL = {"Complete Genome": 1.0, "Chromosome": 0.8, "Scaffold": 0.5, "Contig": 0.3}

    # --- relative importance -> normalised weights that sum to 1 ------------------
    _wraw = {"quality": w_quality.value, "ani": w_ani.value, "exo": w_exo.value,
             "gam": w_gam.value, "nrec": w_nrec.value}
    _wtot = sum(_wraw.values())
    w = ({k: 1.0 / len(_wraw) for k in _wraw} if _wtot <= 0
         else {k: v / _wtot for k, v in _wraw.items()})

    _W_LABELS = [("quality", "genome quality"), ("ani", "ANI to GJ10"),
                 ("exo", "has EXO partner"), ("gam", "has Gam"),
                 ("nrec", "# unique recombinases")]
    _weight_md = mo.md(
        "| criterion | slider | normalised weight |\n|---|---|---|\n"
        + "\n".join(f"| {lbl} | {_wraw[k]:.2f} | **{w[k]:.1%}** |" for k, lbl in _W_LABELS)
        + f"\n| **total** | **{_wtot:.2f}** | **100%** |")

    def _genome_scores():
        gj = [ani_idx[a] for a in GJ10_ACCS if a in ani_idx]
        ani_ref, quality, n_unique = {}, {}, {}
        for _r in meta_df.iter_rows(named=True):
            a = _r["accession"]
            vals = [ANI[ani_idx[a], j] for j in gj if a in ani_idx and np.isfinite(ANI[ani_idx[a], j])]
            ani_ref[a] = max(vals) if vals else np.nan
            quality[a] = _LVL.get(_r.get("level") or "", 0.3)
            n_unique[a] = 0
        for _r in unique_df.iter_rows(named=True):
            g = _r["representative"].split("|")[0]
            n_unique[g] = n_unique.get(g, 0) + 1
        exo = set(operons_df.filter(pl.col("partner_role") == "EXO")["genome"].to_list()) if operons_df.height else set()
        gam = set(operons_df.filter(pl.col("partner_role") == "accessory")["genome"].to_list()) if operons_df.height else set()
        q_n = _minmax(list(quality.values()))
        a_n = _minmax(list(ani_ref.values()))
        n_n = _minmax(list(n_unique.values()))
        rows = []
        for _r in meta_df.iter_rows(named=True):
            a = _r["accession"]
            rows.append({
                "genome": a, "species": _r["species"], "strain": _r["strain"],
                "level": _r["level"], "contigs": _r["contigs"], "is_tiny": _r["is_tiny"],
                "quality": round(q_n(quality[a]), 3),
                "ani_to_ref": None if not np.isfinite(ani_ref[a]) else round(ani_ref[a], 2),
                "ani_norm": round(a_n(ani_ref[a]), 3),
                "has_exo": a in exo, "has_gam": a in gam,
                "n_unique": n_unique.get(a, 0), "n_unique_norm": round(n_n(n_unique.get(a, 0)), 3),
                "score": round(w["quality"] * q_n(quality[a])
                               + w["ani"] * a_n(ani_ref[a])
                               + w["exo"] * float(a in exo)
                               + w["gam"] * float(a in gam)
                               + w["nrec"] * n_n(n_unique.get(a, 0)), 3),
            })
        return pl.DataFrame(rows).sort("score", descending=True)

    genome_scores = _genome_scores()

    _exo_partners = ({f"{_r['genome']}|{_r['ssap_protein']}"
                      for _r in operons_df.filter(pl.col("partner_role") == "EXO").iter_rows(named=True)}
                     if operons_df.height else set())
    recombinase_shortlist = (unique_df
        .with_columns(pl.col("representative").str.split("|").list.first().alias("genome"))
        .join(genome_scores.select(["genome", "score", "ani_to_ref", "has_exo", "has_gam"]),
              on="genome", how="left")
        .with_columns(pl.col("representative").is_in(list(_exo_partners)).alias("has_exo_partner"))
        .sort("score", descending=True))

    _dup_rep = {}
    for _fam, (_lab, _id, _sim, _cov, _recs) in unique_msa.items():
        for _comp in single_linkage(_lab, _id, _cov, dup_id_slider.value, cov_slider.value):
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
        f"Score = Σ (normalised component × weight); weights are normalised to sum to **1.000**. "
        f"**{recombinase_shortlist.height} recombinases** and "
        f"**{system_shortlist.height} deduplicated systems** ranked.")
    mo.vstack([
        mo.md("#### 6d. Weighted ranking"),
        _weight_md,
        mo.ui.tabs({
            "Recombinases (unique SSAP reps)": mo.ui.table(
                recombinase_shortlist.select(
                    ["score", "family", "representative", "n_strain_dupes", "system_group",
                     "is_divergent", "genome", "ani_to_ref", "has_exo_partner"]),
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
def results_genome_eval(
    log_ani,
    log_ani_recompute,
    log_heat_ani,
    log_pcoa,
    mo,
    result,
    sum_ani,
    sum_dup,
    sum_nn,
    sum_rank,
):
    result(mo.accordion({"Results": mo.vstack([log_ani_recompute, log_ani, log_pcoa, sum_ani, sum_nn, sum_dup, sum_rank, log_heat_ani])}), title="Genome evaluation & ranking")
    return


@app.cell
def exports(
    dup_groups_df,
    genome_scores,
    hmm_hits_df,
    meta_df,
    mo,
    operons_df,
    psi_exo_hits_df,
    psi_hits_df,
    recomb_df,
    recomb_loci_df,
    recombinase_shortlist,
    result,
    species_clusters_df,
    system_shortlist,
    unique_df,
):

    def _dl(df, name):
        return mo.download(data=df.write_csv().encode(), filename=name,
                           mimetype="text/csv", label=f"⬇ {name} ({df.height})")

    result(mo.vstack([
        mo.md("## Exports\n\nDownload the **current, filtered** tables as CSV "
              "(they reflect the thresholds set above)."),
        mo.md("**Genomes / ANI**"),
        mo.hstack([_dl(meta_df, "genome_metadata.csv"),
                   _dl(dup_groups_df, "duplicate_groups.csv"),
                   _dl(species_clusters_df, "species_clusters.csv"),
                   _dl(genome_scores, "genome_scores.csv")], justify="start", gap=1),
        mo.md("**Searches**"),
        mo.hstack([_dl(hmm_hits_df, "hmm_hits.csv"),
                   _dl(psi_hits_df, "psiblast_ssap_hits.csv"),
                   _dl(psi_exo_hits_df, "psiblast_exo_hits.csv")], justify="start", gap=1),
        mo.md("**Catalog / results**"),
        mo.hstack([_dl(recomb_df, "recombineering_catalog.csv"),
                   _dl(unique_df, "unique_recombinases.csv"),
                   _dl(recomb_loci_df, "recombining_loci.csv"),
                   _dl(operons_df, "ssap_partner_pairs.csv")], justify="start", gap=1),
        mo.md("**Ranked shortlists**"),
        mo.hstack([_dl(recombinase_shortlist, "recombinase_shortlist.csv"),
                   _dl(system_shortlist, "system_shortlist.csv")], justify="start", gap=1),
    ]))
    return


if __name__ == "__main__":
    app.run()
