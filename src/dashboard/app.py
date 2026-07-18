import os
from datetime import datetime, timedelta, timezone
import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
import streamlit as st

#paths & constants
_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(_ROOT, "data", "raw", "activity.db")

CAP_SECS = 300   # max duration credited to a single window event (5 min)

COLORS = {
    "productive":  "#4ade80",
    "distracting": "#f87171",
    "neutral":     "#94a3b8",
    "unknown":     "#64748b",
    "excluded":    "#475569",
}

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# page setup

st.set_page_config(
    page_title="Procrastination Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

.block-container { padding-top: 1.6rem; padding-bottom: 2rem; }

div[data-testid="stHorizontalBlock"] > div { gap: 0.8rem; }

/* ── metric cards ─────────────────────────────────────────────────────────── */
.card {
    background: #0f172a;
    border-radius: 14px;
    padding: 1.25rem 1.4rem 1rem;
    border: 1px solid #1e293b;
    border-left: 4px solid #334155;
    height: 100%;
}
.card.green  { border-left-color: #4ade80; }
.card.red    { border-left-color: #f87171; }
.card.yellow { border-left-color: #facc15; }
.card.blue   { border-left-color: #60a5fa; }
.card.purple { border-left-color: #a78bfa; }

.card .label {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.5rem;
}
.card .value {
    color: #f1f5f9;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.card .sub {
    color: #475569;
    font-size: 0.77rem;
    margin-top: 0.35rem;
}

/* ── section headings ────────────────────────────────────────────────────── */
.sec-title {
    color: #475569;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.8rem 0 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e293b;
}

/* ── empty state ─────────────────────────────────────────────────────────── */
.empty {
    background: #0a0f1e;
    border: 1px dashed #1e293b;
    border-radius: 10px;
    padding: 2rem 1rem;
    text-align: center;
    color: #334155;
    font-size: 0.88rem;
}

/* ── status pill ─────────────────────────────────────────────────────────── */
.pill {
    display: inline-block;
    padding: 0.15rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.pill.green  { background: #052e16; color: #4ade80; }
.pill.red    { background: #2d0a0a; color: #f87171; }
.pill.yellow { background: #1c1500; color: #facc15; }
.pill.gray   { background: #1e293b; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# shared plotly layout defaults

_CHART_BASE = dict(
    paper_bgcolor="#090e1a",
    plot_bgcolor="#090e1a",
    font=dict(color="#cbd5e1", family="Inter, system-ui, sans-serif"),
)


# timezone helper

_LOCAL_OFFSET = pd.Timedelta(
    seconds=datetime.now(timezone.utc).astimezone().utcoffset().total_seconds()
)

def to_local(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, unit="s") + _LOCAL_OFFSET


# db helpers

def _connect():
    import sqlite3
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def _table_exists(conn, name: str) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone())


# data loaders (cached 60 s)

def _add_duration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["duration"] = (df["ts"].shift(-1) - df["ts"]).clip(upper=CAP_SECS).fillna(0)
    return df


@st.cache_data(ttl=60)
def load_window() -> pd.DataFrame:
    try:
        conn = _connect()
        if not _table_exists(conn, "window_events"):
            conn.close()
            return pd.DataFrame()
        df = pd.read_sql("SELECT ts, app_name, title, category FROM window_events ORDER BY ts", conn)
        conn.close()
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    df["category"] = df["category"].fillna("unknown")
    df["dt"]        = to_local(df["ts"])
    df["date"]      = df["dt"].dt.date
    df["hour"]      = df["dt"].dt.hour
    df["weekday"]   = df["dt"].dt.weekday          # 0 = Monday
    df = _add_duration(df)
    return df


@st.cache_data(ttl=60)
def load_tabs() -> pd.DataFrame:
    try:
        conn = _connect()
        if not _table_exists(conn, "tab_events"):
            conn.close()
            return pd.DataFrame()
        df = pd.read_sql("SELECT ts, domain, title, category FROM tab_events ORDER BY ts", conn)
        conn.close()
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    df["category"] = df["category"].fillna("unknown")
    df["dt"]       = to_local(df["ts"])
    df = _add_duration(df)
    return df


@st.cache_data(ttl=60)
def load_nudges() -> pd.DataFrame:
    try:
        conn = _connect()
        if not _table_exists(conn, "nudge_log"):
            conn.close()
            return pd.DataFrame()
        df = pd.read_sql("SELECT ts, prob, was_helpful, model_version FROM nudge_log ORDER BY ts DESC", conn)
        conn.close()
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    df["dt"] = to_local(df["ts"])
    return df


# UI primitives

def card(col, label: str, value: str, sub: str = "", accent: str = ""):
    col.markdown(f"""
    <div class="card {accent}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def empty_state(msg: str = "Not enough data yet — keep the logger running!"):
    st.markdown(f'<div class="empty">⏳&ensp;{msg}</div>', unsafe_allow_html=True)


def sec(title: str):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)


# section: header

def render_header(df_win: pd.DataFrame):
    now_str = datetime.now().strftime("%A, %d %b %Y · %H:%M")

    # current streak label
    streak_label = ""
    if not df_win.empty:
        last = df_win.iloc[-1]
        cat  = last["category"]
        color_map = {"productive": "green", "distracting": "red", "neutral": "gray"}
        pill_color = color_map.get(cat, "gray")
        mins = int(df_win.groupby(
            (df_win["category"] != df_win["category"].shift()).cumsum()
        )["duration"].transform("sum").iloc[-1] / 60)
        streak_label = f'<span class="pill {pill_color}">{cat.upper()} · {mins}m streak</span>'

    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.2rem">
        <div style="display:flex; align-items:center; gap:14px">
            <span style="font-size:2.2rem; line-height:1"></span>
            <div>
                <h1 style="margin:0; font-size:1.65rem; color:#f8fafc; font-weight:700; line-height:1.1">
                    Procrastination Predictor
                </h1>
                <p style="margin:0.25rem 0 0; color:#475569; font-size:0.83rem">{now_str}</p>
            </div>
        </div>
        <div>{streak_label}</div>
    </div>
    """, unsafe_allow_html=True)


# section: today at a glance

def render_today(df_win: pd.DataFrame, nudges: pd.DataFrame):
    sec("Today at a glance")

    today     = datetime.now().date()
    df_today  = df_win[df_win["date"] == today] if not df_win.empty else pd.DataFrame()
    c1, c2, c3, c4 = st.columns(4)

    if df_today.empty:
        for col, lbl in zip([c1, c2, c3, c4],
                            ["Active time", "Productive", "Distracting", "Nudges today"]):
            card(col, lbl, "—", "no data yet")
        return

    total   = df_today["duration"].sum()
    prod    = df_today[df_today["category"] == "productive"]["duration"].sum()
    dist    = df_today[df_today["category"] == "distracting"]["duration"].sum()
    p_pct   = prod / total if total else 0
    d_pct   = dist / total if total else 0
    tot_min = int(total / 60)

    nudges_today = (
        len(nudges[nudges["dt"].dt.date == today])
        if not nudges.empty else 0
    )

    time_str = f"{tot_min // 60}h {tot_min % 60}m" if tot_min >= 60 else f"{tot_min}m"

    card(c1, "Active time", time_str,
         f"{df_today['app_name'].nunique()} apps used", "blue")
    card(c2, "Productive", f"{p_pct:.0%}",
         f"{int(prod/60)} min in productive apps", "green")
    card(c3, "Distracting", f"{d_pct:.0%}",
         f"{int(dist/60)} min on distracting apps", "red" if d_pct > 0.25 else "")
    card(c4, "Nudges fired", str(nudges_today),
         "today", "yellow" if nudges_today > 0 else "")


# section: activity heatmap

def render_heatmap(df_win: pd.DataFrame):
    sec("Procrastination heatmap — hour × day of week")

    if df_win.empty or df_win["duration"].sum() < 600:
        empty_state("Heatmap needs a few more hours of data to be meaningful.")
        return

    # distracting seconds per (hour, weekday) cell
    total_by = df_win.groupby(["hour", "weekday"])["duration"].sum()
    dist_by  = (
        df_win[df_win["category"] == "distracting"]
        .groupby(["hour", "weekday"])["duration"].sum()
    )
    heat = (dist_by / total_by).fillna(0).rename("ratio").reset_index()

    pivot = (
        heat.pivot(index="hour", columns="weekday", values="ratio")
        .reindex(index=range(24), columns=range(7))
        .fillna(0)
    )

    hour_labels = [
        ("12am" if h == 0 else f"{h}am" if h < 12 else "12pm" if h == 12 else f"{h-12}pm")
        for h in range(24)
    ]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=DAYS,
        y=hour_labels,
        colorscale=[
            [0.00, "#090e1a"],
            [0.25, "#1a2744"],
            [0.55, "#7c2020"],
            [1.00, "#ef4444"],
        ],
        zmin=0, zmax=1,
        hovertemplate="%{y} on %{x}<br>Distracting: %{z:.0%}<extra></extra>",
        colorbar=dict(
            title=dict(text="Distraction %", font=dict(color="#64748b", size=11)),
            tickformat=".0%",
            tickfont=dict(color="#64748b", size=10),
            thickness=12,
            len=0.85,
        ),
    ))

    fig.update_layout(
        **_CHART_BASE,
        height=440,
        margin=dict(l=55, r=20, t=10, b=35),
        yaxis=dict(autorange="reversed", gridcolor="#0f172a", tickfont=dict(size=10)),
        xaxis=dict(gridcolor="#0f172a", side="top", tickfont=dict(size=12)),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each cell = average fraction of that hour spent on distracting apps/sites. Darker red = more distracting.")


# section: 24-hour category timeline

def render_timeline(df_win: pd.DataFrame):
    sec("Last 24 hours")

    if df_win.empty:
        empty_state()
        return

    cutoff  = datetime.now() - timedelta(hours=24)
    df_24   = df_win[df_win["dt"] >= pd.Timestamp(cutoff)].copy()

    if len(df_24) < 5:
        empty_state("Not enough activity in the last 24 hours.")
        return

    df_24["bucket"] = df_24["dt"].dt.floor("30min")
    grouped = df_24.groupby(["bucket", "category"])["duration"].sum().reset_index()
    totals  = grouped.groupby("bucket")["duration"].sum().rename("total")
    grouped = grouped.join(totals, on="bucket")
    grouped["ratio"] = grouped["duration"] / grouped["total"]

    fig = go.Figure()
    for cat in ["productive", "neutral", "distracting", "unknown"]:
        g = grouped[grouped["category"] == cat]
        c = COLORS[cat]
        fig.add_trace(go.Scatter(
            x=g["bucket"], y=g["ratio"],
            name=cat.capitalize(),
            stackgroup="one",
            fillcolor="rgba(74,222,128,0.55)" if cat == "productive" else "rgba(148,163,184,0.55)" if cat == "neutral" else "rgba(248,113,113,0.55)" if cat == "distracting" else "rgba(100,116,139,0.55)",
            line=dict(color=c, width=0.5),
            hovertemplate=f"{cat}: %{{y:.0%}}<extra></extra>",
        ))

    fig.update_layout(
        **_CHART_BASE,
        height=290,
        margin=dict(l=50, r=10, t=15, b=35),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11)),
        xaxis=dict(gridcolor="#0f172a", tickformat="%H:%M", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#1e293b", tickformat=".0%", range=[0, 1],
                   tickfont=dict(size=10)),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


# section: top distractors

def _bar(df: pd.DataFrame, x_col: str, label: str):
    if df.empty:
        empty_state()
        return

    top = (
        df[df["category"] == "distracting"]
        .groupby(x_col)["duration"].sum()
        .sort_values(ascending=True)
        .tail(8)
        .reset_index()
    )
    top["minutes"] = (top["duration"] / 60).round(1)
    top[x_col] = top[x_col].str.replace(r"\.exe$", "", regex=True)

    if top.empty:
        empty_state(f"No distracting {label} logged yet — nice!")
        return

    fig = go.Figure(go.Bar(
        x=top["minutes"],
        y=top[x_col],
        orientation="h",
        marker=dict(
            color=top["minutes"],
            colorscale=[[0, "#7c2020"], [1, "#ef4444"]],
            showscale=False,
        ),
        hovertemplate="%{y}: %{x} min<extra></extra>",
    ))

    fig.update_layout(
        **_CHART_BASE,
        height=280,
        margin=dict(l=10, r=20, t=10, b=35),
        xaxis=dict(gridcolor="#1e293b", title="minutes", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11)),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_distractors(df_win: pd.DataFrame, df_tab: pd.DataFrame):
    sec("Biggest time sinks")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            "<p style='color:#94a3b8; font-size:0.82rem; margin-bottom:0.4rem'>📱 &nbsp;Apps</p>",
            unsafe_allow_html=True,
        )
        _bar(df_win, "app_name", "apps")

    with c2:
        st.markdown(
            "<p style='color:#94a3b8; font-size:0.82rem; margin-bottom:0.4rem'>🌐 &nbsp;Sites (Chrome)</p>",
            unsafe_allow_html=True,
        )
        if df_tab.empty:
            empty_state("No tab events yet — is the Chrome extension loaded?")
        else:
            _bar(df_tab, "domain", "sites")


# section: 14-day trend 

def render_trend(df_win: pd.DataFrame):
    sec("14-day productivity trend")

    if df_win.empty:
        empty_state()
        return

    cutoff  = datetime.now().date() - timedelta(days=14)
    df_14   = df_win[df_win["date"] >= cutoff].copy()

    if df_14["date"].nunique() < 2:
        empty_state("Need at least 2 days of data for a trend — check back tomorrow.")
        return

    daily   = df_14.groupby(["date", "category"])["duration"].sum().reset_index()
    totals  = daily.groupby("date")["duration"].sum().rename("total")
    daily   = daily.join(totals, on="date")
    daily["ratio"] = daily["duration"] / daily["total"]

    def daily_ratio(cat):
        return (
            daily[daily["category"] == cat][["date", "ratio"]]
            .rename(columns={"ratio": cat})
        )

    trend = (
        daily_ratio("productive")
        .merge(daily_ratio("distracting"), on="date", how="outer")
        .fillna(0)
        .sort_values("date")
    )
    trend["date"] = pd.to_datetime(trend["date"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trend["date"], y=trend["productive"],
        name="Productive",
        line=dict(color="#4ade80", width=2.5),
        fill="tozeroy", fillcolor="rgba(74,222,128,0.13)",
        hovertemplate="Productive: %{y:.0%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=trend["date"], y=trend["distracting"],
        name="Distracting",
        line=dict(color="#f87171", width=2.5),
        fill="tozeroy", fillcolor="rgba(248,113,113,0.13)",
        hovertemplate="Distracting: %{y:.0%}<extra></extra>",
    ))

    fig.update_layout(
        **_CHART_BASE,
        height=270,
        margin=dict(l=50, r=20, t=15, b=35),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11)),
        xaxis=dict(gridcolor="#1e293b", tickformat="%b %d", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#1e293b", tickformat=".0%", range=[0, 1],
                   tickfont=dict(size=10)),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


# section: nudge history 

def render_nudges(nudges: pd.DataFrame):
    sec("Nudge history")

    if nudges.empty:
        empty_state("No nudges fired yet — predictor is warming up or threshold not crossed.")
        return

    total     = len(nudges)
    helpful   = int(nudges["was_helpful"].fillna(0).sum())
    h_rate    = helpful / total if total else 0

    c1, c2 = st.columns([1, 2])

    with c1:
        card(st, "Total nudges", str(total),
             f"{helpful} acknowledged · {h_rate:.0%} helpfulness", "yellow")

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(h_rate * 100),
            number=dict(suffix="%", font=dict(color="#f1f5f9", size=38)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#334155",
                          tickfont=dict(color="#475569", size=10)),
                bar=dict(color="#facc15"),
                bgcolor="#0f172a",
                bordercolor="#1e293b",
                steps=[
                    dict(range=[0, 40],  color="#0f172a"),
                    dict(range=[40, 70], color="#1a2033"),
                    dict(range=[70, 100], color="#142514"),
                ],
            ),
            title=dict(text="Helpfulness rate",
                       font=dict(color="#64748b", size=12)),
        ))
        fig.update_layout(
            paper_bgcolor="#090e1a",
            height=200,
            margin=dict(l=20, r=20, t=40, b=5),
            font=dict(color="#cbd5e1", family="Inter, system-ui"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        display = nudges.head(30).copy()
        display["Time"]       = display["dt"].dt.strftime("%b %d  %H:%M")
        display["Confidence"] = display["prob"].apply(lambda p: f"{p:.0%}")
        display["Response"]   = display["was_helpful"].map(
            {1: "✅  On it", 0: "❌  Dismissed"}
        ).fillna("—")
        display["Mode"] = display["model_version"].str.upper()

        st.dataframe(
            display[["Time", "Confidence", "Response", "Mode"]].reset_index(drop=True),
            use_container_width=True,
            height=290,
            hide_index=True,
        )


# main layout

def main():
    df_win  = load_window()
    df_tab  = load_tabs()
    nudges  = load_nudges()

    render_header(df_win)

    col_btn, _ = st.columns([1, 11])
    with col_btn:
        if st.button("↻ Refresh", help="Force-reload all data from the database"):
            st.cache_data.clear()
            st.rerun()

    render_today(df_win, nudges)

    st.markdown("---")

    left, right = st.columns([3, 2], gap="large")
    with left:
        render_heatmap(df_win)
    with right:
        render_timeline(df_win)

    st.markdown("---")
    render_distractors(df_win, df_tab)

    st.markdown("---")

    left2, right2 = st.columns([3, 2], gap="large")
    with left2:
        render_trend(df_win)
    with right2:
        render_nudges(nudges)


main()
