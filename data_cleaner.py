# =====================================================================
#  DATA CLEANER  -  Automated Data Cleaning Studio (Gradio, Colab-ready)
#  Fixes: missingness, duplicates, outliers, skewness, kurtosis
#  Outputs: cleaned CSV + full step-by-step report (HTML + CSV log)
# =====================================================================
# Colab:  !pip install -q gradio pandas numpy scipy scikit-learn plotly
#         then run this file's code in a cell (or %run data_cleaner_claude_ui.py)

import io, os, html, tempfile, warnings, inspect
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import gradio as gr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Cleaning engine
# ---------------------------------------------------------------------
class CleaningLog:
    def __init__(self):
        self.rows = []

    def add(self, stage, column, action, detail, before=None, after=None):
        self.rows.append({
            "Stage": stage, "Column": column, "Action": action,
            "Detail": detail,
            "Before": "" if before is None else before,
            "After": "" if after is None else after,
        })

    def df(self):
        return pd.DataFrame(self.rows, columns=["Stage", "Column", "Action", "Detail", "Before", "After"])


def _fmt(x):
    if isinstance(x, (float, np.floating)):
        return f"{x:.4g}"
    return str(x)


def snapshot(df):
    num = df.select_dtypes(include=np.number)
    sk = num.skew().abs().mean() if not num.empty else np.nan
    ku = num.kurt().abs().mean() if not num.empty else np.nan
    return {
        "rows": len(df), "cols": df.shape[1],
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "mean_abs_skew": sk, "mean_abs_kurt": ku,
    }


def tidy_basics(df, log):
    """Strip header whitespace, trim strings, unify blank-like tokens to NaN, coerce numeric-looking text."""
    old_cols = list(df.columns)
    df.columns = [str(c).strip() for c in df.columns]
    renamed = [(o, n) for o, n in zip(old_cols, df.columns) if o != n]
    if renamed:
        log.add("Prep", "headers", "Trimmed whitespace", f"{len(renamed)} header(s) cleaned")

    blanks = {"", " ", "na", "n/a", "nan", "null", "none", "-", "--", "?", "missing", "undefined"}
    converted_total = 0
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype("string").str.strip()
            mask = s.str.lower().isin(blanks)
            if mask.sum():
                converted_total += int(mask.sum())
                s = s.mask(mask, pd.NA)
            df[c] = s.astype(object).where(s.notna(), np.nan)
            # numeric coercion if >=90% of non-null values parse as numbers
            nn = df[c].dropna()
            if len(nn):
                parsed = pd.to_numeric(nn.astype(str).str.replace(",", "", regex=False), errors="coerce")
                if parsed.notna().mean() >= 0.90:
                    df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
                    log.add("Prep", c, "Type coercion", "Text column converted to numeric")
    if converted_total:
        log.add("Prep", "all", "Blank tokens -> NaN", f"{converted_total} placeholder value(s) (e.g. 'N/A', '?', '-') treated as missing")
    return df


def drop_duplicates(df, log):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    log.add("Duplicates", "all", "Removed exact duplicate rows" if removed else "No duplicates found",
            f"{removed} row(s) removed", before, len(df))
    return df


def handle_missing(df, log, col_drop_thr, row_drop_thr, num_strategy, cat_strategy):
    # 1. drop very sparse columns
    miss_pct = df.isna().mean() * 100
    to_drop = miss_pct[miss_pct > col_drop_thr].index.tolist()
    for c in to_drop:
        log.add("Missingness", c, "Dropped column", f"{miss_pct[c]:.1f}% missing > {col_drop_thr}% threshold",
                f"{miss_pct[c]:.1f}% missing", "removed")
    df = df.drop(columns=to_drop)

    # 2. drop very sparse rows
    if df.shape[1]:
        row_pct = df.isna().mean(axis=1) * 100
        bad = row_pct > row_drop_thr
        if bad.sum():
            log.add("Missingness", "rows", "Dropped rows",
                    f"{int(bad.sum())} row(s) with > {row_drop_thr}% missing values", len(df), len(df) - int(bad.sum()))
            df = df.loc[~bad].reset_index(drop=True)

    # 3. impute
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = [c for c in df.columns if c not in num_cols]

    if num_strategy == "KNN (k=5)" and len(num_cols) > 1 and df[num_cols].isna().any().any():
        try:
            from sklearn.impute import KNNImputer
            miss_before = df[num_cols].isna().sum()
            df[num_cols] = KNNImputer(n_neighbors=5).fit_transform(df[num_cols])
            for c in num_cols:
                if miss_before[c]:
                    log.add("Missingness", c, "KNN imputation", f"{int(miss_before[c])} value(s) imputed (k=5)",
                            f"{int(miss_before[c])} missing", 0)
        except Exception as e:
            log.add("Missingness", "numeric", "KNN unavailable", f"Fell back to median ({e})")
            num_strategy = "Median"

    for c in num_cols:
        n = int(df[c].isna().sum())
        if not n:
            continue
        if num_strategy == "Mean":
            val = df[c].mean(); name = "mean"
        elif num_strategy == "Smart (auto)":
            # skewed -> median, symmetric -> mean
            sk = df[c].skew()
            if abs(sk) > 0.5:
                val = df[c].median(); name = f"median (skew={sk:.2f})"
            else:
                val = df[c].mean(); name = f"mean (skew={sk:.2f})"
        else:
            val = df[c].median(); name = "median"
        if pd.isna(val):
            val = 0; name = "0 (column all-NaN)"
        df[c] = df[c].fillna(val)
        log.add("Missingness", c, f"Imputed with {name}", f"{n} value(s) filled with {_fmt(val)}", f"{n} missing", 0)

    for c in cat_cols:
        n = int(df[c].isna().sum())
        if not n:
            continue
        if cat_strategy == "Mode":
            m = df[c].mode(dropna=True)
            val = m.iloc[0] if len(m) else "Unknown"
            name = "mode"
        else:
            val = "Unknown"; name = "'Unknown' label"
        df[c] = df[c].fillna(val)
        log.add("Missingness", c, f"Imputed with {name}", f"{n} value(s) filled with '{val}'", f"{n} missing", 0)
    return df


def handle_outliers(df, log, method, iqr_k, z_thr, action):
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if df[c].nunique() > 5]
    if not num_cols or method == "Skip":
        log.add("Outliers", "all", "Skipped", "Outlier handling disabled or no continuous numeric columns")
        return df
    mask_any = pd.Series(False, index=df.index)
    for c in num_cols:
        s = df[c]
        if method == "IQR":
            q1, q3 = s.quantile(.25), s.quantile(.75)
            iqr = q3 - q1
            lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
            desc = f"IQR x{iqr_k}"
        else:
            mu, sd = s.mean(), s.std()
            if not sd or np.isnan(sd):
                continue
            lo, hi = mu - z_thr * sd, mu + z_thr * sd
            desc = f"|Z| > {z_thr}"
        m = (s < lo) | (s > hi)
        n = int(m.sum())
        if not n:
            continue
        if action == "Cap (winsorize)":
            df[c] = s.clip(lo, hi)
            log.add("Outliers", c, "Capped to bounds", f"{n} outlier(s) [{desc}] clipped to [{_fmt(lo)}, {_fmt(hi)}]",
                    f"min={_fmt(s.min())} max={_fmt(s.max())}", f"min={_fmt(df[c].min())} max={_fmt(df[c].max())}")
        elif action == "Remove rows":
            mask_any |= m
            log.add("Outliers", c, "Flagged for removal", f"{n} outlier row(s) [{desc}] outside [{_fmt(lo)}, {_fmt(hi)}]")
        else:  # median replace
            med = s.median()
            df.loc[m, c] = med
            log.add("Outliers", c, "Replaced with median", f"{n} outlier(s) [{desc}] -> {_fmt(med)}")
    if action == "Remove rows" and mask_any.sum():
        b = len(df)
        df = df.loc[~mask_any].reset_index(drop=True)
        log.add("Outliers", "rows", "Removed outlier rows", f"{b - len(df)} row(s) dropped (union across columns)", b, len(df))
    return df


def handle_shape(df, log, skew_thr, kurt_thr, do_skew, do_kurt):
    """Fix skewness with Box-Cox / Yeo-Johnson / log1p / sqrt; fix heavy-tailed kurtosis with
    Yeo-Johnson + percentile winsorizing."""
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if df[c].nunique() > 10]
    if not num_cols:
        log.add("Skew/Kurtosis", "all", "Skipped", "No continuous numeric columns")
        return df

    for c in num_cols:
        s = df[c].astype(float)
        sk0, ku0 = stats.skew(s), stats.kurtosis(s)   # excess kurtosis
        need_skew = do_skew and abs(sk0) > skew_thr
        need_kurt = do_kurt and abs(ku0) > kurt_thr
        if not (need_skew or need_kurt):
            log.add("Skew/Kurtosis", c, "OK", f"skew={sk0:.2f}, kurtosis={ku0:.2f} within limits", f"{sk0:.2f}/{ku0:.2f}", f"{sk0:.2f}/{ku0:.2f}")
            continue

        candidates = {}
        mn = s.min()
        sign = 1 if sk0 > 0 else -1
        # candidate transforms
        try:
            if sign > 0:
                shift = 0 if mn >= 0 else -mn
                candidates["log1p"] = np.log1p(s + shift)
                candidates["sqrt"] = np.sqrt(s + shift)
                if mn + shift > 0 or (s + shift).min() >= 0:
                    pos = s + shift + 1e-9
                    bc, _ = stats.boxcox(pos)
                    candidates["Box-Cox"] = pd.Series(bc, index=s.index)
            else:  # left skew -> reflect then log
                refl = (s.max() + 1) - s
                candidates["reflect+log1p"] = np.log1p(refl)
                candidates["square"] = s ** 2
            yj, _ = stats.yeojohnson(s)
            candidates["Yeo-Johnson"] = pd.Series(yj, index=s.index)
        except Exception:
            pass

        if not candidates:
            continue

        def score(t):
            return abs(stats.skew(t)) + 0.25 * abs(stats.kurtosis(t))

        best_name = min(candidates, key=lambda k: score(candidates[k]))
        best = candidates[best_name]
        sk1, ku1 = stats.skew(best), stats.kurtosis(best)

        # Only accept if it truly improves things
        if score(best) < score(s):
            df[c] = best.values
            log.add("Skew/Kurtosis", c, f"Applied {best_name}",
                    f"skew {sk0:.2f} -> {sk1:.2f} | kurtosis {ku0:.2f} -> {ku1:.2f}", f"{sk0:.2f}/{ku0:.2f}", f"{sk1:.2f}/{ku1:.2f}")
            sk0, ku0 = sk1, ku1
            s = df[c].astype(float)
        else:
            log.add("Skew/Kurtosis", c, "No transform helped", f"Kept original (skew={sk0:.2f}, kurtosis={ku0:.2f})")

        # Residual heavy tails -> percentile winsorizing
        if do_kurt and abs(stats.kurtosis(s)) > kurt_thr:
            for p in (0.005, 0.01, 0.025):
                lo, hi = s.quantile(p), s.quantile(1 - p)
                tmp = s.clip(lo, hi)
                if abs(stats.kurtosis(tmp)) <= kurt_thr:
                    break
            ku_b = stats.kurtosis(s)
            df[c] = tmp
            log.add("Skew/Kurtosis", c, f"Tail winsorized ({p*100:.1f}% / {100-p*100:.1f}%)",
                    f"kurtosis {ku_b:.2f} -> {stats.kurtosis(tmp):.2f}", f"{ku_b:.2f}", f"{stats.kurtosis(tmp):.2f}")
    return df


def run_pipeline(file, col_drop_thr, row_drop_thr, num_strategy, cat_strategy,
                 out_method, iqr_k, z_thr, out_action,
                 do_skew, do_kurt, skew_thr, kurt_thr, progress=gr.Progress()):
    if file is None:
        raise gr.Error("Upload a CSV file first.")
    path = file if isinstance(file, str) else file.name
    progress(0.05, desc="Reading CSV")
    df = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
            break
        except Exception:
            continue
    if df is None or df.empty:
        raise gr.Error("Could not parse the CSV (or it is empty).")

    raw = df.copy()
    before = snapshot(df)
    log = CleaningLog()

    progress(0.2, desc="Normalising headers & types")
    df = tidy_basics(df, log)
    stage0 = df.copy()   # tidy 'before' snapshot used by the charts
    progress(0.35, desc="Removing duplicates")
    df = drop_duplicates(df, log)
    progress(0.5, desc="Fixing missing values")
    df = handle_missing(df, log, col_drop_thr, row_drop_thr, num_strategy, cat_strategy)
    progress(0.65, desc="Treating outliers")
    df = handle_outliers(df, log, out_method, iqr_k, z_thr, out_action)
    progress(0.8, desc="Fixing skewness & kurtosis")
    df = handle_shape(df, log, skew_thr, kurt_thr, do_skew, do_kurt)
    progress(0.9, desc="Final duplicate sweep")
    df = drop_duplicates(df, log)

    after = snapshot(df)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmpdir = tempfile.mkdtemp()
    csv_path = os.path.join(tmpdir, f"cleaned_{ts}.csv")
    log_path = os.path.join(tmpdir, f"cleaning_log_{ts}.csv")
    rep_path = os.path.join(tmpdir, f"cleaning_report_{ts}.html")
    df.to_csv(csv_path, index=False)
    logdf = log.df()
    logdf.to_csv(log_path, index=False)

    report_html = build_report(before, after, logdf, raw, df)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(f"<html><head><meta charset='utf-8'><link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500&display=swap' rel='stylesheet'><style>{REPORT_CSS}</style></head><body style='background:#FAF9F5;padding:24px'>{report_html}</body></html>")

    progress(1.0, desc="Done")
    progress(0.96, desc="Drawing charts")
    choices = [c for c in df.select_dtypes(include=np.number).columns if c in stage0.columns]
    first = None
    if choices:
        def _sk(c):
            v = pd.to_numeric(stage0[c], errors="coerce").dropna()
            return abs(stats.skew(v)) if len(v) > 3 else 0
        first = max(choices, key=_sk)
    figs = build_all_charts(stage0, df, first)
    return (report_html, df.head(200), logdf, [csv_path, log_path, rep_path],
            before_after_dist(raw, df), (stage0, df.copy()),
            gr.Dropdown(choices=choices, value=first), *figs)


def before_after_dist(raw, clean):
    """Return a small summary dataframe of skew/kurtosis before vs after."""
    rows = []
    for c in clean.select_dtypes(include=np.number).columns:
        if c in raw.columns and pd.api.types.is_numeric_dtype(raw[c]):
            r = raw[c].dropna()
            if len(r) > 3:
                rows.append({"Column": c,
                             "Skew before": round(stats.skew(r), 3), "Skew after": round(stats.skew(clean[c]), 3),
                             "Kurtosis before": round(stats.kurtosis(r), 3), "Kurtosis after": round(stats.kurtosis(clean[c]), 3)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------
def card(label, b, a, fmt="{:,}"):
    def f(x):
        return "-" if (isinstance(x, float) and np.isnan(x)) else fmt.format(x)
    return (f"<div class='kpi'><div class='kl'>{label}</div>"
            f"<div class='kv'><span class='b'>{f(b)}</span><span class='ar'>&#10132;</span><span class='a'>{f(a)}</span></div></div>")


def build_report(before, after, logdf, raw, clean):
    stage_icons = {}
    kpis = "".join([
        card("ROWS", before["rows"], after["rows"]),
        card("COLUMNS", before["cols"], after["cols"]),
        card("MISSING CELLS", before["missing"], after["missing"]),
        card("DUPLICATE ROWS", before["duplicates"], after["duplicates"]),
        card("AVG |SKEW|", before["mean_abs_skew"], after["mean_abs_skew"], "{:.2f}"),
        card("AVG |KURTOSIS|", before["mean_abs_kurt"], after["mean_abs_kurt"], "{:.2f}"),
    ])
    stages_html = ""
    for stage, g in logdf.groupby("Stage", sort=False):
        items = ""
        for _, r in g.iterrows():
            ba = f"<span class='ba'>{html.escape(str(r['Before']))} &#10132; {html.escape(str(r['After']))}</span>" if r["Before"] != "" else ""
            items += (f"<li><b class='col'>{html.escape(str(r['Column']))}</b> "
                      f"<span class='act'>{html.escape(str(r['Action']))}</span><br>"
                      f"<span class='det'>{html.escape(str(r['Detail']))}</span> {ba}</li>")
        stages_html += (f"<div class='stage'><h3>{stage_icons.get(stage, '&#10059;')} {html.escape(stage)} "
                        f"<span class='cnt'>{len(g)} step(s)</span></h3><ul>{items}</ul></div>")
    return (f"<div class='rep'><h2>Cleaning report</h2>"
            f"<p class='sub'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &middot; "
            f"{len(logdf)} actions logged</p><div class='kpis'>{kpis}</div>{stages_html}</div>")


# ---------------------------------------------------------------------
# Charts (Plotly, Claude palette)
# ---------------------------------------------------------------------
import plotly.graph_objects as go
from plotly.subplots import make_subplots

C_BEFORE, C_AFTER = "#B0AEA5", "#C6613F"
C_BLUE, C_GREEN, C_INK, C_GRID = "#6A9BCC", "#788C5D", "#141413", "#EEECE2"


def _style(fig, title, h=290, legend=True):
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", y=0.96,
                   font=dict(family="Source Serif 4, Georgia, serif", size=16, color=C_INK)),
        height=h, margin=dict(l=46, r=16, t=56, b=42),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", size=11, color="#3D3D3A"),
        showlegend=legend,
        legend=dict(orientation="h", y=1.14, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E8E6DC", font=dict(family="Inter", color=C_INK)),
        bargap=0.28,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#DDD9CE", ticks="outside", tickcolor="#DDD9CE")
    fig.update_yaxes(gridcolor=C_GRID, zeroline=False, linecolor="rgba(0,0,0,0)")
    return fig


def _empty(title, msg, h=290):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(family="Inter", size=13, color="#73726C"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, title, h, legend=False)


def _num_cols(df):
    return [c for c in df.select_dtypes(include=np.number).columns]


def chart_distribution(stage0, clean, col):
    t = "Distribution: before vs after"
    if not col or col not in clean.columns or col not in stage0.columns:
        return _empty(t, "Pick a numeric column")
    b = pd.to_numeric(stage0[col], errors="coerce").dropna()
    a = clean[col].dropna()
    if len(b) < 3 or len(a) < 3:
        return _empty(t, "Not enough data")
    sb, sa = stats.skew(b), stats.skew(a)
    kb, ka = stats.kurtosis(b), stats.kurtosis(a)
    fig = go.Figure()
    fig.add_histogram(x=b, histnorm="probability density", name="Before", marker_color=C_BEFORE,
                      opacity=0.75, nbinsx=40)
    fig.add_histogram(x=a, histnorm="probability density", name="After", marker_color=C_AFTER,
                      opacity=0.65, nbinsx=40)
    fig.update_layout(barmode="overlay")
    fig.add_annotation(x=0.99, y=0.97, xref="paper", yref="paper", xanchor="right", yanchor="top", showarrow=False,
                       align="right", font=dict(size=11, color="#5E5D59"),
                       text=f"skew {sb:.2f} \u2192 <b>{sa:.2f}</b><br>kurtosis {kb:.2f} \u2192 <b>{ka:.2f}</b>")
    fig.update_yaxes(title_text="density")
    fig.update_xaxes(title_text=str(col))
    return _style(fig, f"{t}  \u00b7  {col}")


def chart_box(stage0, clean, col):
    t = "Spread and outliers"
    if not col or col not in clean.columns or col not in stage0.columns:
        return _empty(t, "Pick a numeric column")
    b = pd.to_numeric(stage0[col], errors="coerce").dropna()
    a = clean[col].dropna()
    fig = go.Figure()
    fig.add_box(x=b, name="Before", marker_color=C_BEFORE, line_color="#8A8880", boxmean=True,
                fillcolor="rgba(176,174,165,.35)", marker=dict(size=4, opacity=.6))
    fig.add_box(x=a, name="After", marker_color=C_AFTER, line_color=C_AFTER, boxmean=True,
                fillcolor="rgba(198,97,63,.22)", marker=dict(size=4, opacity=.6))
    fig.update_xaxes(title_text=str(col), showgrid=True, gridcolor=C_GRID)
    fig.update_yaxes(showgrid=False)
    return _style(fig, f"{t}  \u00b7  {col}", legend=False)


def chart_missing(stage0, clean):
    t = "Missing values by column (%)"
    pct_b = (stage0.isna().mean() * 100).sort_values(ascending=False)
    pct_b = pct_b[pct_b > 0].head(12)
    if pct_b.empty:
        return _empty(t, "No missing values in the original data")
    cols = pct_b.index.tolist()[::-1]
    after = [float(clean[c].isna().mean() * 100) if c in clean.columns else None for c in cols]
    dropped = [c not in clean.columns for c in cols]
    fig = go.Figure()
    fig.add_bar(y=cols, x=[pct_b[c] for c in cols], orientation="h", name="Before", marker_color=C_BEFORE)
    fig.add_bar(y=cols, x=[0 if v is None else v for v in after], orientation="h", name="After",
                marker_color=C_AFTER,
                text=["dropped" if d else "" for d in dropped], textposition="outside", cliponaxis=False,
                textfont=dict(color="#73726C", size=10))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="% of rows", showgrid=True, gridcolor=C_GRID)
    fig.update_yaxes(showgrid=False)
    return _style(fig, t, h=max(290, 90 + 34 * len(cols)))


def _iqr_count(s):
    s = s.dropna()
    if len(s) < 5:
        return 0
    q1, q3 = s.quantile(.25), s.quantile(.75)
    i = q3 - q1
    return int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum())


def chart_outliers(stage0, clean):
    t = "Outliers per column (1.5\u00d7IQR rule)"
    cols = [c for c in _num_cols(clean) if c in stage0.columns and clean[c].nunique() > 5]
    if not cols:
        return _empty(t, "No continuous numeric columns")
    nb = {c: _iqr_count(pd.to_numeric(stage0[c], errors="coerce")) for c in cols}
    cols = sorted(cols, key=lambda c: nb[c], reverse=True)[:12]
    if all(nb[c] == 0 for c in cols):
        return _empty(t, "No outliers detected in the original data")
    fig = go.Figure()
    fig.add_bar(x=cols, y=[nb[c] for c in cols], name="Before", marker_color=C_BEFORE)
    fig.add_bar(x=cols, y=[_iqr_count(clean[c]) for c in cols], name="After", marker_color=C_AFTER)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="count")
    fig.update_xaxes(tickangle=-30)
    return _style(fig, t)


def chart_shape(stage0, clean):
    t = "Skewness and kurtosis"
    cols = [c for c in _num_cols(clean) if c in stage0.columns and clean[c].nunique() > 10]
    if not cols:
        return _empty(t, "No continuous numeric columns", h=300)
    rows = []
    for c in cols:
        b = pd.to_numeric(stage0[c], errors="coerce").dropna()
        if len(b) > 3:
            rows.append((c, abs(stats.skew(b)), abs(stats.skew(clean[c])),
                         abs(stats.kurtosis(b)), abs(stats.kurtosis(clean[c]))))
    if not rows:
        return _empty(t, "Not enough data", h=300)
    rows = sorted(rows, key=lambda r: r[1], reverse=True)[:12]
    names = [r[0] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("|Skewness|", "|Excess kurtosis|"), horizontal_spacing=0.09)
    fig.add_bar(x=names, y=[r[1] for r in rows], name="Before", marker_color=C_BEFORE, row=1, col=1)
    fig.add_bar(x=names, y=[r[2] for r in rows], name="After", marker_color=C_AFTER, row=1, col=1)
    fig.add_bar(x=names, y=[r[3] for r in rows], name="Before", marker_color=C_BEFORE, showlegend=False, row=1, col=2)
    fig.add_bar(x=names, y=[r[4] for r in rows], name="After", marker_color=C_AFTER, showlegend=False, row=1, col=2)
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickangle=-30)
    for a in fig.layout.annotations:
        a.font = dict(family="Inter", size=12, color="#5E5D59")
        a.xanchor = "left"
        a.x = a.x - 0.2
    return _style(fig, t, h=310)


def chart_corr(stage0, clean):
    t = "Correlation structure"
    cols = [c for c in _num_cols(clean) if c in stage0.columns][:10]
    if len(cols) < 2:
        return _empty(t, "Need at least two numeric columns", h=330)
    b = stage0[cols].apply(pd.to_numeric, errors="coerce").corr()
    a = clean[cols].corr()
    show_txt = len(cols) <= 7
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Before", "After"), horizontal_spacing=0.12)
    for i, m in enumerate((b, a), start=1):
        fig.add_trace(go.Heatmap(z=m.values, x=cols, y=cols, zmin=-1, zmax=1, coloraxis="coloraxis",
                                 text=m.round(2).values, texttemplate="%{text}" if show_txt else None,
                                 textfont=dict(size=10),
                                 hovertemplate="%{y} \u00d7 %{x}: %{z:.2f}<extra></extra>"), row=1, col=i)
    fig.update_layout(coloraxis=dict(colorscale=[[0, C_BLUE], [0.5, "#FAF9F5"], [1, C_AFTER]], cmin=-1, cmax=1,
                                     colorbar=dict(thickness=10, len=0.8, outlinewidth=0, tickfont=dict(size=10))))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-35)
    for a_ in fig.layout.annotations:
        a_.font = dict(family="Inter", size=12, color="#5E5D59")
    return _style(fig, t, h=360, legend=False)


def build_dist_box(stage0, clean, col):
    return chart_distribution(stage0, clean, col), chart_box(stage0, clean, col)


def build_all_charts(stage0, clean, col):
    d, b = build_dist_box(stage0, clean, col)
    return d, b, chart_missing(stage0, clean), chart_outliers(stage0, clean), \
        chart_shape(stage0, clean), chart_corr(stage0, clean)


def on_pick(state, col):
    if not state:
        return _empty("Distribution: before vs after", "Run the cleaner first"), _empty("Spread and outliers", "Run the cleaner first")
    stage0, clean = state
    return build_dist_box(stage0, clean, col)


REPORT_CSS = r"""
.rep{font-family:'Inter',-apple-system,'Segoe UI',sans-serif;color:#141413;font-size:13px}
.rep h2{font-family:'Source Serif 4',Georgia,serif;font-weight:500;font-size:22px;margin:0;color:#141413}
.rep .sub{color:#73726C;margin:2px 0 12px;font-size:12px}
.kpis{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px;margin-bottom:12px}
@media (max-width:1100px){.kpis{grid-template-columns:repeat(3,minmax(0,1fr))}}
.kpi{background:#FFFFFF;border:1px solid #E8E6DC;border-radius:12px;padding:9px 11px}
.kl{font-size:10px;letter-spacing:.06em;color:#73726C;text-transform:uppercase}
.kv{font-family:'Source Serif 4',Georgia,serif;font-size:17px;margin-top:3px;white-space:nowrap}
.kv .b{color:#8A8880}.kv .a{color:#C6613F;font-weight:500}.kv .ar{color:#B0AEA5;margin:0 5px;font-size:13px}
.stage{background:#FFFFFF;border:1px solid #E8E6DC;border-radius:12px;margin:8px 0;padding:8px 14px}
.stage h3{font-family:'Source Serif 4',Georgia,serif;font-weight:500;color:#141413;font-size:15px;margin:2px 0 4px}
.cnt{font-family:'Inter',sans-serif;font-size:10px;color:#73726C;margin-left:8px;background:#F0EEE6;padding:1px 8px;border-radius:20px}
.stage ul{list-style:none;padding:0;margin:0}
.stage li{padding:6px 0;border-bottom:1px solid #F0EEE6;line-height:1.4}
.stage li:last-child{border:none}
.col{color:#3D3D3A;background:#F0EEE6;padding:1px 7px;border-radius:6px;font-weight:500}
.act{color:#C6613F;font-weight:500;margin-left:6px}
.det{color:#5E5D59}.ba{color:#8A8880;font-size:11px;margin-left:6px}
.empty{text-align:center;padding:70px 20px;color:#73726C;font-family:'Inter',sans-serif}
.empty .big{font-family:'Source Serif 4',Georgia,serif;font-size:26px;color:#141413;margin:6px 0}
.empty .spark{color:#D97757;font-size:34px}
"""

# ---------------------------------------------------------------------
# Claude-style theme + compact single-screen UI
# ---------------------------------------------------------------------
FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500&display=swap');\n"

CSS = FONT_IMPORT + r"""
.gradio-container{
  --body-background-fill:#FAF9F5; --background-fill-primary:#FFFFFF; --background-fill-secondary:#F5F4ED;
  --block-background-fill:#FFFFFF; --block-border-color:#E8E6DC; --block-border-width:1px; --block-radius:12px;
  --block-label-background-fill:transparent; --block-label-text-color:#73726C; --block-title-text-color:#3D3D3A;
  --block-padding:8px 10px; --layout-gap:8px; --form-gap-width:8px; --spacing-lg:8px;
  --input-padding:6px 10px; --input-text-size:13px; --text-md:13px; --text-sm:12px;
  --body-text-color:#141413; --body-text-color-subdued:#73726C;
  --input-background-fill:#FFFFFF; --input-border-color:#DDD9CE; --input-border-color-focus:#C6613F;
  --input-shadow:none; --input-shadow-focus:0 0 0 3px rgba(198,97,63,.15);
  --border-color-primary:#E8E6DC; --border-color-accent:#C6613F; --color-accent:#C6613F; --color-accent-soft:#F6E3DA;
  --button-primary-background-fill:#C6613F; --button-primary-background-fill-hover:#B5532F;
  --button-primary-text-color:#FFFFFF; --button-primary-border-color:#C6613F;
  --slider-color:#C6613F; --table-even-background-fill:#FFFFFF; --table-odd-background-fill:#FAF9F5;
  --table-border-color:#E8E6DC; --table-text-color:#141413; --shadow-drop:none; --shadow-drop-lg:none;
  --checkbox-label-background-fill:transparent; --checkbox-label-border-color:transparent;
  background:#FAF9F5!important;color:#141413!important;
  font-family:'Inter',-apple-system,'Segoe UI',sans-serif!important;font-size:14px;
  max-width:100%!important;padding:0 20px!important;height:100vh;overflow:hidden}
html,body{height:100%;overflow:hidden;background:#FAF9F5!important}
footer{display:none!important}

/* header */
#hdr{background:transparent!important;border:none!important;box-shadow:none!important;padding:0!important}
#app-header{display:flex;align-items:center;gap:10px;height:56px}
#app-header .spark{color:#D97757;font-size:26px;line-height:1}
#app-header .ttl{font-family:'Source Serif 4',Georgia,serif;font-size:22px;font-weight:500;color:#141413}
#app-header .tag{color:#73726C;font-size:13px;margin-left:6px}
#css-inject{display:none!important}

/* single-screen layout */
#main{height:calc(100vh - 68px);flex-wrap:nowrap!important;gap:16px;align-items:stretch}
#side{flex:0 0 340px!important;min-width:340px!important;max-width:340px;height:100%;overflow-y:auto;gap:10px}
#content{height:100%;min-width:0;overflow:hidden}
.pane{max-height:calc(100vh - 150px);overflow:auto}
@media (max-width:900px){
  html,body,.gradio-container{height:auto;overflow:auto}
  #main{flex-direction:column!important;height:auto}
  #side{flex:1 1 auto!important;min-width:0!important;max-width:100%;height:auto}
  .pane{max-height:none}
}

/* surfaces */
.block,.form,.panel{box-shadow:none!important}
#side{background:#F5F4ED;border:1px solid #E8E6DC;border-radius:16px;padding:12px!important}
#side .block{background:transparent}
#side .form{background:transparent;border:none}
label span,.gr-block-label{color:#73726C!important;font-weight:500;font-size:12px!important}
#upload{background:#FFFFFF!important;border:1.5px dashed #CFCBBE!important;border-radius:12px!important}
#upload .wrap,#upload button.center{min-height:84px!important;padding:8px!important}

/* inputs */
input:not([type=checkbox]):not([type=radio]):not([type=range]),textarea,select{
  background:#FFFFFF!important;color:#141413!important;border:1px solid #DDD9CE!important;border-radius:8px!important}
input:not([type=checkbox]):not([type=radio]):not([type=range]):focus{border-color:#C6613F!important}
input[type=range]{accent-color:#C6613F}
input[type=checkbox]{
  -webkit-appearance:none!important;appearance:none!important;
  width:18px!important;height:18px!important;min-width:18px;cursor:pointer;position:relative;
  background:#FFFFFF!important;border:1.5px solid #B0AEA5!important;border-radius:5px!important;
  transition:.15s;pointer-events:auto!important;z-index:2}
input[type=checkbox]:hover{border-color:#C6613F!important}
input[type=checkbox]:checked{background:#C6613F!important;border-color:#C6613F!important}
input[type=checkbox]:checked::after{
  content:"\2713";position:absolute;left:3px;top:-1px;color:#FFFFFF;font-weight:700;font-size:13px;line-height:18px}
label:has(input[type=checkbox]){cursor:pointer}

/* primary button */
button.primary,#run{background:#C6613F!important;color:#FFFFFF!important;border:none!important;
  border-radius:10px!important;font-family:'Inter',sans-serif!important;font-weight:500!important;
  font-size:14px!important;height:40px;transition:background .15s}
button.primary:hover,#run:hover{background:#B5532F!important}

/* tabs */
button[role=tab],.tab-nav button{font-family:'Inter',sans-serif!important;font-weight:500;font-size:13px!important;
  color:#73726C!important;padding:6px 12px!important;background:transparent!important;border:none!important}
button[role=tab][aria-selected=true],.tab-nav button.selected{color:#141413!important;
  border-bottom:2px solid #C6613F!important}
.tab-wrapper,.tab-container,.tab-nav{border-color:#E8E6DC!important}
.tabitem{background:transparent!important;border:none!important;padding:8px 0 0!important}
#side .tabitem{padding:8px 0 0!important}

/* tables */
table{color:#141413!important;font-size:12px!important}
thead th{background:#F0EEE6!important;color:#3D3D3A!important;font-weight:600}
""" + REPORT_CSS

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.orange, neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#FAF9F5", body_background_fill_dark="#FAF9F5",
    block_background_fill="#FFFFFF", block_background_fill_dark="#FFFFFF",
    body_text_color="#141413", body_text_color_dark="#141413",
    block_border_color="#E8E6DC", block_border_color_dark="#E8E6DC",
    input_background_fill="#FFFFFF", input_background_fill_dark="#FFFFFF",
    button_primary_background_fill="#C6613F", button_primary_background_fill_dark="#C6613F",
    button_primary_background_fill_hover="#B5532F", button_primary_background_fill_hover_dark="#B5532F",
    button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#FFFFFF",
)

EMPTY = ("<div class='empty'><div class='spark'>&#10059;</div>"
         "<div class='big'>Upload a CSV to get started</div>"
         "<div>Choose your settings on the left, then click <b>Clean dataset</b>.</div></div>")

# Gradio 6 moved theme/css to launch(); older versions take them in Blocks().
_NEW_API = "css" in inspect.signature(gr.Blocks.launch).parameters
_blocks_kwargs = dict(title="Data Cleaner")
_launch_kwargs = dict(share=True, debug=False)
if _NEW_API:
    _launch_kwargs.update(theme=theme, css=CSS)
else:
    _blocks_kwargs.update(theme=theme, css=CSS)

with gr.Blocks(**_blocks_kwargs) as demo:
    gr.HTML("<style>" + CSS + "</style>", elem_id="css-inject")
    gr.HTML("<div id='app-header'><span class='spark'>&#10059;</span><span class='ttl'>Data Cleaner</span>"
            "<span class='tag'>Fix missing values, duplicates, outliers, skew and kurtosis &mdash; "
            "then download the cleaned CSV and report.</span></div>", elem_id="hdr")

    with gr.Row(elem_id="main", equal_height=False):
        # ---------------- sidebar ----------------
        with gr.Column(elem_id="side", scale=0, min_width=340):
            f_in = gr.File(label="CSV file", file_types=[".csv", ".tsv", ".txt"], type="filepath", elem_id="upload")
            with gr.Tabs(elem_id="opts"):
                with gr.Tab("Missing"):
                    with gr.Row():
                        col_thr = gr.Slider(10, 100, 60, step=5, label="Drop column if % missing >", min_width=0)
                        row_thr = gr.Slider(10, 100, 70, step=5, label="Drop row if % missing >", min_width=0)
                    with gr.Row():
                        num_s = gr.Dropdown(["Smart (auto)", "Median", "Mean", "KNN (k=5)"], value="Smart (auto)",
                                            label="Numeric fill", min_width=0)
                        cat_s = gr.Dropdown(["Mode", "Unknown label"], value="Mode", label="Categorical fill", min_width=0)
                with gr.Tab("Outliers"):
                    with gr.Row():
                        o_m = gr.Dropdown(["IQR", "Z-score", "Skip"], value="IQR", label="Method", min_width=0)
                        o_a = gr.Dropdown(["Cap (winsorize)", "Replace with median", "Remove rows"],
                                          value="Cap (winsorize)", label="Treatment", min_width=0)
                    with gr.Row():
                        iqr_k = gr.Slider(1.0, 3.0, 1.5, step=0.1, label="IQR multiplier", min_width=0)
                        z_t = gr.Slider(2.0, 5.0, 3.0, step=0.1, label="Z threshold", min_width=0)
                with gr.Tab("Distribution"):
                    with gr.Row():
                        d_s = gr.Checkbox(True, label="Fix skewness", min_width=0)
                        d_k = gr.Checkbox(True, label="Fix kurtosis", min_width=0)
                    with gr.Row():
                        s_t = gr.Slider(0.25, 2.0, 0.75, step=0.05, label="|Skew| limit", min_width=0)
                        k_t = gr.Slider(0.5, 10.0, 3.0, step=0.25, label="|Kurtosis| limit", min_width=0)
            run = gr.Button("Clean dataset", variant="primary", elem_id="run")

        # ---------------- content ----------------
        with gr.Column(elem_id="content", scale=1):
            with gr.Tabs():
                with gr.Tab("Report"):
                    with gr.Column(elem_classes="pane"):
                        rep = gr.HTML(EMPTY)
                with gr.Tab("Charts"):
                    with gr.Column(elem_classes="pane"):
                        st = gr.State(None)
                        col_pick = gr.Dropdown([], label="Column for distribution and box plot", elem_id="colpick")
                        with gr.Row():
                            p_dist = gr.Plot(show_label=False)
                            p_box = gr.Plot(show_label=False)
                        with gr.Row():
                            p_miss = gr.Plot(show_label=False)
                            p_out = gr.Plot(show_label=False)
                        p_sk = gr.Plot(show_label=False)
                        p_corr = gr.Plot(show_label=False)
                with gr.Tab("Cleaned data"):
                    with gr.Column(elem_classes="pane"):
                        prev = gr.Dataframe(label="Preview (first 200 rows)", interactive=False, wrap=False)
                with gr.Tab("Skew / kurtosis"):
                    with gr.Column(elem_classes="pane"):
                        sk_tbl = gr.Dataframe(label="Before vs after", interactive=False)
                with gr.Tab("Action log"):
                    with gr.Column(elem_classes="pane"):
                        log_tbl = gr.Dataframe(label="Every step taken", interactive=False, wrap=True)
                with gr.Tab("Downloads"):
                    with gr.Column(elem_classes="pane"):
                        outf = gr.File(label="Cleaned CSV  |  Log CSV  |  HTML report", file_count="multiple")

    run.click(run_pipeline,
              inputs=[f_in, col_thr, row_thr, num_s, cat_s, o_m, iqr_k, z_t, o_a, d_s, d_k, s_t, k_t],
              outputs=[rep, prev, log_tbl, outf, sk_tbl, st, col_pick, p_dist, p_box, p_miss, p_out, p_sk, p_corr])
    col_pick.change(on_pick, inputs=[st, col_pick], outputs=[p_dist, p_box])

if __name__ == "__main__":
    demo.queue().launch(**_launch_kwargs)
