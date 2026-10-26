# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.17.0",
#     "pandas>=2.2",
#     "altair>=5.4",
# ]
# ///

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def imports():
    import altair as alt
    import pandas as pd

    return alt, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Recombinase discovery in _Xanthobacter_ — results explorer

        A read-only companion to `recombinase_workflow.py`. That notebook needs
        BLAST+, HMMER, MAFFT, skani, the NCBI `datasets` CLI and ~5 GB of genome
        FASTA to produce these numbers; this one needs nothing but the tables it
        ships with (≈120 kB), so it runs anywhere: locally, on
        [molab](https://molab.marimo.io), in the browser via WebAssembly, or as a
        static page.

        **What is here:** Pfam/PSI-BLAST hit tables, the deduplicated bench
        shortlist, SSAP↔exonuclease operon calls, ANI and pairwise-identity
        matrices, and the reference-phage cross-references. Every table carries
        its provenance (source file + SHA-256) in the **Data provenance** tab.

        To regenerate the bundle after re-running the workflow:

        ```bash
        cd Recombinase_Search
        python share/build_bundle.py      # writes share/data/*.csv + manifest.json
        ```
        """
    )
    return


@app.cell
def data_access(mo):
    """Locate the bundle in all four places this notebook is expected to run."""
    import json
    import urllib.request
    from pathlib import Path

    # Set this once you push the repo; used as the fallback when the notebook is
    # opened from molab/WebAssembly with no repo checkout beside it.
    GH_REPO = "nakrasian/recombineering"
    GH_REF = "main"
    SUBDIR = "Recombinase_Search"

    def _read_bytes(path_or_url) -> bytes:
        """Read a local path or an HTTPS URL (WASM/static-host case)."""
        try:
            p = Path(str(path_or_url))
            if p.exists():
                return p.read_bytes()
        except OSError:
            pass
        req = urllib.request.Request(
            str(path_or_url), headers={"User-Agent": "marimo-results-explorer"}
        )
        with urllib.request.urlopen(req) as fh:
            return fh.read()

    def _sources(name: str):
        """Candidate locations, most-specific first.

        Local checkout → whatever `mo.notebook_location()` points at (a directory
        when served by marimo, the page URL under WebAssembly) → GitHub raw.
        """
        try:
            here = Path(__file__).resolve().parent
            yield here / "share" / "data" / name
        except NameError:
            pass
        try:
            loc = mo.notebook_location()
        except Exception:
            loc = None
        if loc is not None:
            try:
                yield loc / "share" / "data" / name
            except TypeError:
                yield f"{str(loc).rstrip('/')}/share/data/{name}"
        if "/" in GH_REPO and "your-org" not in GH_REPO:
            yield (
                f"https://raw.githubusercontent.com/{GH_REPO}/{GH_REF}/{SUBDIR}"
                f"/share/data/{name}"
            )

    def load_json(name: str):
        errors = []
        for src in _sources(name):
            try:
                return json.loads(_read_bytes(src).decode("utf-8")), str(src)
            except Exception as exc:  # keep trying, then report all attempts
                errors.append(f"{src}: {type(exc).__name__}")
        raise FileNotFoundError("could not load " + name + " — tried\n  " + "\n  ".join(errors))

    def load_csv(name: str, **kwargs):
        import io

        import pandas as pd

        errors = []
        for src in _sources(name):
            try:
                return pd.read_csv(io.BytesIO(_read_bytes(src)), **kwargs), str(src)
            except Exception as exc:
                errors.append(f"{src}: {type(exc).__name__}")
        raise FileNotFoundError("could not load " + name + " — tried\n  " + "\n  ".join(errors))

    return GH_REPO, load_csv, load_json


@app.cell
def manifest(load_json, mo):
    try:
        MANIFEST, _manifest_src = load_json("manifest.json")
    except FileNotFoundError as exc:
        mo.stop(True, mo.callout(mo.md(f"**Bundle not found.**\n\n```\n{exc}\n```"),
                                 kind="danger"))

    META = {t["file"]: t for t in MANIFEST["tables"]}
    DATASET = MANIFEST.get("dataset", "dataset")
    mo.md(
        f"Loaded **{len(META)} tables** from the `{DATASET}` bundle "
        f"({sum(t['bytes'] for t in MANIFEST['tables']) / 1024:.0f} kB)."
    )
    return DATASET, META


@app.cell
def tables(load_csv, mo, pd):
    """Load every CSV once; a broken table becomes empty instead of fatal."""
    NAMES = [
        "shortlist", "species_counts", "genome_counts", "hmm_hits",
        "psiblast_ssap", "psiblast_exo", "operons", "operon_pairs",
        "ani_matrix", "identity_matrix", "ref_phage_recomb", "ref_phage_collection",
        "phage_hits_by_type", "refseq_virus_vs_phastest", "genomes",
    ]
    T, LOG = {}, []
    for name in NAMES:
        try:
            df, src = load_csv(f"{name}.csv", dtype=str, keep_default_na=False)
            T[name] = df
            LOG.append(f"{name}: {len(df)} rows")
        except FileNotFoundError as exc:
            T[name] = pd.DataFrame()
            LOG.append(f"{name}: MISSING ({exc})")

    def num(df, *cols):
        """Coerce columns to numeric, leaving unparseable cells as NaN."""
        out = df.copy()
        for c in cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    mo.accordion({"Load log": mo.md("  \n".join(LOG))})
    return T, num


@app.cell
def kpis(DATASET, T, mo):
    def _n(key: str, col: str | None = None):
        df = T.get(key)
        if df is None or df.empty:
            return 0
        return int(df[col].nunique()) if col else len(df)

    _operon_genomes = T["operon_pairs"]["genome"].nunique() if not T["operon_pairs"].empty else 0
    _families = sorted(set(T["operons"]["family"])) if not T["operons"].empty else []

    mo.hstack(
        [
            mo.stat(_n("genomes"), label="genomes", caption=f"dataset `{DATASET}`"),
            mo.stat(_n("operons"), label="SSAP / exonuclease proteins", caption="with coordinates"),
            mo.stat(_operon_genomes, label="genomes with an operon", caption="SSAP ↔ EXO ≤ 5 kb"),
            mo.stat(_n("shortlist"), label="shortlisted groups", caption="bench candidates"),
            mo.stat(len(_families), label="families", caption=", ".join(_families) or "—"),
        ],
        widths="equal",
        wrap=True,
    )
    return


@app.cell
def tab_shortlist(T, alt, mo):
    _df = T["shortlist"]
    if _df.empty:
        shortlist_view = mo.callout(mo.md("`shortlist.csv` missing from the bundle."), kind="warn")
    else:
        _long = (
            _df.assign(
                redundant=_df["redundant_variants(>=90%)"].str.count(";").add(1).where(
                    _df["redundant_variants(>=90%)"].str.len() > 0, 0
                ),
                distinct=_df["distinct_members(70-90%)"].str.count(";").add(1).where(
                    _df["distinct_members(70-90%)"].str.len() > 0, 0
                ),
            )
            .melt(
                id_vars=["group", "family", "representative", "ngens"],
                value_vars=["redundant", "distinct"],
                var_name="relation",
                value_name="members",
            )
        )

        _chart = (
            alt.Chart(_long)
            .mark_bar()
            .encode(
                x=alt.X("group:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-60)),
                y=alt.Y("members:Q", title="variant proteins", stack=None),
                color=alt.Color("relation:N", title=None),
                column=alt.Column("family:N", title=None, header=alt.Header(labelFontSize=11)),
                tooltip=["group:N", "family:N", "representative:N", "ngens:Q", "relation:N", "members:Q"],
            )
            .properties(width=260, height=220)
        )

        shortlist_view = mo.vstack(
            [
                mo.md(
                    "Deduplicated groups of recombinases. `ngens` is how many genomes "
                    "carry the group; `redundant` are near-identical variants (≥90% id) "
                    "and `distinct` are the 70–90% members worth testing separately."
                ),
                mo.ui.altair_chart(_chart),
                mo.accordion(
                    {
                        "Table — shortlist": mo.ui.table(_df, selection=None, pagination=True, page_size=20),
                        "Download": mo.download(
                            _df.to_csv(index=False).encode(),
                            filename="shortlist.csv",
                            label="shortlist.csv",
                        ),
                    }
                ),
            ]
        )
    return (shortlist_view,)


@app.cell
def tab_genomes(T, alt, num, pd, mo):
    _genomes = T["genomes"]
    _counts = T["genome_counts"]
    if _counts.empty:
        genomes_view = mo.callout(mo.md("`genome_counts.csv` missing."), kind="warn")
    else:
        _num = num(_counts, "RecT", "ERF", "Rad52", "exonuclease")
        _long = _num.melt(
            id_vars=["genome", "species", "label", "system_status"],
            value_vars=[c for c in ["RecT", "ERF", "Rad52", "exonuclease"] if c in _num.columns],
            var_name="family",
            value_name="hits",
        )
        _bygenome = _long.groupby("genome", as_index=False)["hits"].sum()

        _bar = (
            alt.Chart(_bygenome)
            .transform_bin("bin", field="hits", bin=alt.Bin(maxbins=14))
            .mark_bar()
            .encode(
                x=alt.X("bin:Q", title="hits per genome"),
                y=alt.Y("count():Q", title="genomes"),
                tooltip=[alt.Tooltip("count():Q", title="genomes")],
            )
            .properties(height=200, title="How many hits most genomes carry")
        )

        _species = (
            alt.Chart(_long.groupby(["species", "family"], as_index=False)["hits"].sum())
            .mark_bar()
            .encode(
                y=alt.Y("species:N", sort="-x", title=None),
                x=alt.X("sum(hits):Q", title="hits"),
                color=alt.Color("family:N", title=None),
                tooltip=["species:N", "family:N", alt.Tooltip("sum(hits):Q", title="hits")],
            )
            .properties(height=max(220, 16 * _long["species"].nunique()),
                        title="Hits by species and family")
        )

        _status = (
            _counts.assign(
                system_status=_counts["system_status"].replace("", "no operon").fillna("no operon")
            )
            .groupby("system_status", as_index=False)
            .size()
            .rename(columns={"size": "genomes"})
        )
        _status_chart = (
            alt.Chart(_status)
            .mark_arc(innerRadius=45)
            .encode(
                theta=alt.Theta("genomes:Q"),
                color=alt.Color("system_status:N", title=None),
                tooltip=["system_status:N", "genomes:Q"],
            )
            .properties(height=200, title="Operon status")
        )

        # ANI heatmap: melt the wide matrix, drop the diagonal, keep one triangle
        _ani = T["ani_matrix"]
        _ani_chart = mo.md("")
        if not _ani.empty:
            _m = _ani.set_index("accession").apply(lambda s: pd.to_numeric(s, errors="coerce"))
            _stack = _m.stack().rename("ani").reset_index()
            _stack.columns = ["a", "b", "ani"]
            _stack = _stack[_stack["a"] < _stack["b"]]
            _order = list(_m.index)[:60]  # a 121² grid is unreadable; show a corner
            _sub = _stack[_stack["a"].isin(_order) & _stack["b"].isin(_order)]
            _ani_chart = (
                alt.Chart(_sub)
                .mark_rect()
                .encode(
                    x=alt.X("b:N", sort=list(_m.index), title=None, axis=alt.Axis(labels=False, ticks=False)),
                    y=alt.Y("a:N", sort=list(_m.index), title=None, axis=alt.Axis(labels=False, ticks=False)),
                    color=alt.Color("ani:Q", scale=alt.Scale(scheme="viridis", domainMin=80), title="ANI %"),
                    tooltip=["a:N", "b:N", alt.Tooltip("ani:Q", format=".2f")],
                )
                .properties(height=420, width=440,
                            title="Pairwise ANI (skani), first 60 genomes")
            )

        genomes_view = mo.vstack(
            [
                mo.hstack([_bar, _status_chart], widths="equal", wrap=True),
                _species,
                mo.accordion(
                    {
                        "ANI heatmap": _ani_chart,
                        "Per-genome table": mo.ui.table(_counts, selection=None, pagination=True, page_size=15),
                        "Genome metadata": mo.ui.table(_genomes, selection=None, pagination=True, page_size=15),
                    }
                ),
            ]
        )
    return (genomes_view,)


@app.cell
def operon_picker(T, mo):
    _feats = T["operons"]
    _genomes = sorted(_feats["genome"].unique()) if not _feats.empty else []
    op_genome_pick = mo.ui.dropdown(
        options=_genomes,
        value=_genomes[0] if _genomes else None,
        label="Browse one genome's proteins",
    )
    return (op_genome_pick,)


@app.cell
def tab_operons(T, alt, num, mo, op_genome_pick):
    _pairs = num(T["operon_pairs"], "gap_bp", "ssap_start", "ssap_end")
    _feats = T["operons"]
    if _pairs.empty:
        operons_view = mo.callout(mo.md("No SSAP↔exonuclease pairs in this bundle."), kind="warn")
    else:
        _gap = (
            alt.Chart(_pairs)
            .mark_bar()
            .encode(
                x=alt.X("gap_bp:Q", title="intergenic gap (bp)", bin=alt.Bin(step=250)),
                y=alt.Y("count():Q", title="pairs"),
                color=alt.Color("relation:N", title="same strand?"),
                tooltip=[alt.Tooltip("count():Q", title="pairs")],
            )
            .properties(height=220, title="SSAP ↔ exonuclease spacing")
        )

        _pair_table = (
            alt.Chart(_pairs)
            .mark_circle(size=90, opacity=0.85)
            .encode(
                x=alt.X("ssap_start:Q", title="SSAP start (bp)", axis=alt.Axis(format="~s")),
                y=alt.Y("gap_bp:Q", title="gap (bp)"),
                color=alt.Color("ssap_family:N", title="SSAP family"),
                shape=alt.Shape("relation:N", title="same strand?"),
                tooltip=["genome:N", "ssap_family:N", "ssap_protein:N", "exo_protein:N",
                         "gap_bp:Q", "relation:N"],
            )
            .properties(height=220, title="Each pair, by position")
        )

        _detail = _feats[_feats["genome"] == op_genome_pick.value] if op_genome_pick.value else _feats

        operons_view = mo.vstack(
            [
                mo.md(
                    "**SSAP** = single-strand annealing protein (RecT / ERF / Rad52); "
                    "**EXO** = the exonuclease it must be paired with for dsDNA "
                    "recombination. `relation` says whether the pair shares a strand — "
                    "same-strand pairs are the cleaner operon candidates."
                ),
                mo.hstack([_gap, _pair_table], widths="equal", wrap=True),
                mo.ui.table(_pairs.sort_values("gap_bp"), selection=None, pagination=True, page_size=12),
                op_genome_pick,
                mo.ui.table(_detail, selection=None, pagination=True, page_size=10),
            ]
        )
    return (operons_view,)


@app.cell
def tab_phages(T, alt, num, mo):
    _ph = num(T["ref_phage_collection"], "blastn_regions", "blastn_sum_bitscore",
              "blastn_max_bitscore", "blastn_max_pident", "blastn_max_coverage",
              "phastest_regions", "phastest_cds_hits", "phastest_sum_neglog10e",
              "phastest_num_genomes", "RecT", "ERF", "Rad52", "Sak", "Gam", "YqaJ")
    if _ph.empty:
        phages_view = mo.callout(mo.md("`ref_phage_collection.csv` missing."), kind="warn")
    else:
        _scatter = (
            alt.Chart(_ph.dropna(subset=["blastn_max_coverage"]))
            .mark_circle(size=80, opacity=0.8)
            .encode(
                x=alt.X("blastn_max_coverage:Q", title="BLASTN max coverage of phage genome"),
                y=alt.Y("phastest_sum_neglog10e:Q", title="PHASTEST support (−log₁₀ e)"),
                color=alt.Color("host_type:N", title="phage host type"),
                size=alt.Size("blastn_regions:Q", title="BLASTN regions",
                              scale=alt.Scale(range=[30, 400])),
                tooltip=["name:N", "accession:N", "host_type:N", "both_methods:N",
                         "blastn_max_pident:Q", "recomb_system:N"],
            )
            .properties(height=300, title="Two independent lines of evidence agree on the right phages")
        )

        _ramp = num(T["ref_phage_collection"],
                    "RecT", "ERF", "Rad52", "Sak", "Gam", "YqaJ").melt(
            id_vars=["name", "host_type"],
            value_vars=["RecT", "ERF", "Rad52", "Sak", "Gam", "YqaJ"],
            var_name="family", value_name="hits",
        )
        _ramp_chart = (
            alt.Chart(_ramp.groupby(["host_type", "family"], as_index=False)["hits"].sum())
            .mark_rect()
            .encode(
                x=alt.X("family:N", title=None),
                y=alt.Y("host_type:N", title=None),
                color=alt.Color("sum(hits):Q", title="hits", scale=alt.Scale(scheme="magma")),
                tooltip=["host_type:N", "family:N", alt.Tooltip("sum(hits):Q", title="hits")],
            )
            .properties(height=180, title="Recombinase families encoded, by phage host type")
        )

        phages_view = mo.vstack(
            [
                mo.hstack([_scatter, _ramp_chart], widths=[2, 1], wrap=True),
                mo.accordion(
                    {
                        "Reference phage panel (both methods)": mo.ui.table(
                            T["ref_phage_collection"], selection=None, pagination=True, page_size=12
                        ),
                        "Per-phage Xanthobacter hits": mo.ui.table(
                            T["ref_phage_recomb"], selection=None, pagination=True, page_size=12
                        ),
                        "PHASTEST hits by host type": mo.ui.table(
                            T["phage_hits_by_type"], selection=None, pagination=True, page_size=12
                        ),
                        "RefSeq virus cross-reference": mo.ui.table(
                            T["refseq_virus_vs_phastest"], selection=None, pagination=True, page_size=12
                        ),
                    }
                ),
            ]
        )
    return (phages_view,)


@app.cell
def raw_picker(META, mo):
    raw_table_pick = mo.ui.dropdown(
        options={f"{t['file']} — {t['title']}": t["file"] for t in META.values()},
        label="Inspect a table",
    )
    return (raw_table_pick,)


@app.cell
def tab_raw(META, T, pd, mo, raw_table_pick):
    """Manifest browser: what is in the bundle, where it came from, download it."""
    _rows = []
    for t in META.values():
        _rows.append({
            "file": t["file"],
            "title": t["title"],
            "rows": t["rows"],
            "cols": t["cols"],
            "KiB": round(t["bytes"] / 1024, 1),
            "source": t["source"],
            "sha256": t["sha256"][:16] + "…",
            "description": t["description"],
        })
    _manifest_df = pd.DataFrame(_rows)

    _name = raw_table_pick.value.replace(".csv", "") if raw_table_pick.value else None
    _view = T.get(_name, pd.DataFrame()) if _name else pd.DataFrame()

    raw_view = mo.vstack(
        [
            mo.md(
                "Every table below is a normalized copy of a workflow artifact "
                "(see `source`). Empty or stale artifacts are excluded; the "
                "workflow's absolute local paths are stripped. Full hashes are in "
                "`share/data/manifest.json`."
            ),
            mo.ui.table(_manifest_df, selection=None, pagination=True, page_size=20),
            raw_table_pick,
            mo.ui.table(_view, selection=None, pagination=True, page_size=15) if not _view.empty else mo.md(""),
            mo.download(
                _view.to_csv(index=False).encode() if not _view.empty else b"",
                filename=f"{_name}.csv" if _name else "table.csv",
                label=f"download {_name}.csv" if _name else "download",
            ) if _name else mo.md(""),
        ]
    )
    return (raw_view,)


@app.cell
def layout(genomes_view, operons_view, phages_view, raw_view, shortlist_view, mo):
    mo.ui.tabs(
        {
            "Shortlist": shortlist_view,
            "Genomes & ANI": genomes_view,
            "Operon pairs": operons_view,
            "Reference phages": phages_view,
            "Data provenance": raw_view,
        }
    )
    return


@app.cell(hide_code=True)
def footer(GH_REPO, mo):
    mo.md(
        f"""
        ---

        ### Reproducing this

        The numbers here come from `recombinase_workflow.py`, which needs
        `blast+`, `hmmer`, `mafft`, `skani`/`fastani` and the NCBI `datasets` CLI
        plus the 121-genome input set. That environment is not available in a
        browser, which is exactly why this notebook only reads the distilled
        tables.

        ```bash
        cd Recombinase_Search
        marimo edit recombinase_workflow.py      # the search (needs the CLIs)
        python share/build_bundle.py             # distill results -> share/data/
        marimo edit recombinase_results_explorer.py
        ```

        * Source notebook: [`recombinase_workflow.py`](./recombinase_workflow.py)
        * Bundle manifest: [`share/data/manifest.json`](./share/data/manifest.json)
        * Repository: `{GH_REPO}`

        When reusing these results, cite the bundle commit — the SHA-256 in the
        provenance tab pins the exact tables.
        """
    )
    return


if __name__ == "__main__":
    app.run()
