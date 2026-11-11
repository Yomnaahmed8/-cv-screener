import glob
import html
import os
import re

import pandas as pd
import pdfplumber
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CV Screener", layout="wide", page_icon="📄")

CSS = """<style>
.block-container {padding-top: 1.5rem; max-width: 1200px;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
.hero {display: flex; align-items: center; justify-content: space-between;
background: linear-gradient(135deg, #0a1226 0%, #0d364a 100%);
border: 1px solid rgba(45, 212, 191, 0.25);
padding: 26px 34px; border-radius: 18px; margin-bottom: 24px;}
.hero h1 {color: #ffffff; margin: 0; padding: 0; font-size: 2.2rem;}
.hero p {color: #2dd4bf; margin: 6px 0 4px 0; font-size: 1.1rem; font-weight: 600;}
.hero small {color: #b9c6d6;}
.hero svg {flex-shrink: 0; margin-left: 20px;}
[data-testid="stMetric"] {background: rgba(148, 163, 184, 0.10);
border: 1px solid rgba(148, 163, 184, 0.25);
border-radius: 14px; padding: 16px 18px;}
.card {background: rgba(148, 163, 184, 0.10);
border: 1px solid rgba(148, 163, 184, 0.25);
border-radius: 16px; padding: 18px 14px; text-align: center;}
.card.first {border-color: rgba(45, 212, 191, 0.6);
box-shadow: 0 0 24px rgba(45, 212, 191, 0.15);}
.medal {font-size: 1.7rem; line-height: 1;}
.avatar {border-radius: 50%; display: flex; align-items: center;
justify-content: center; color: #ffffff; font-weight: 700; margin: 8px auto;}
.cname {font-weight: 700; font-size: 1.05rem;}
.ctitle {opacity: 0.7; font-size: 0.9rem;}
.cmeta {opacity: 0.7; font-size: 0.85rem;}
.exp-head {display: flex; align-items: center; gap: 14px; margin-bottom: 8px;}
.exp-head .avatar {margin: 0;}
.chip {display: inline-block; padding: 3px 12px; margin: 3px 6px 3px 0;
border-radius: 999px; font-size: 0.85rem; font-weight: 600;}
.chip.ok {background: rgba(16, 185, 129, 0.18); color: #34d399;
border: 1px solid rgba(16, 185, 129, 0.35);}
.chip.miss {background: rgba(239, 68, 68, 0.15); color: #f87171;
border: 1px solid rgba(239, 68, 68, 0.35);}
.chip.none {background: rgba(148, 163, 184, 0.18); color: #94a3b8;}
.chip-title {font-weight: 700; margin-top: 10px; opacity: 0.85;}
.foot {text-align: center; opacity: 0.55; font-size: 0.85rem; margin-top: 30px;}
</style>"""

HERO = """<div class="hero"><div><h1>CV Screener</h1>
<p>Rank CVs against a job description in seconds</p>
<small>A screening aid only: a person should make the final decision.</small></div>
<svg viewBox="0 0 260 110" width="260" height="110">
<g stroke="#2dd4bf" stroke-opacity="0.4" stroke-width="1">
<line x1="20" y1="80" x2="70" y2="40"/><line x1="70" y1="40" x2="125" y2="70"/>
<line x1="125" y1="70" x2="180" y2="25"/><line x1="125" y1="70" x2="170" y2="95"/>
<line x1="180" y1="25" x2="235" y2="55"/><line x1="170" y1="95" x2="235" y2="55"/>
<line x1="70" y1="40" x2="110" y2="12"/><line x1="110" y1="12" x2="180" y2="25"/>
</g>
<g fill="#2dd4bf">
<circle cx="20" cy="80" r="4"/><circle cx="70" cy="40" r="5"/>
<circle cx="125" cy="70" r="6"/><circle cx="180" cy="25" r="5"/>
<circle cx="170" cy="95" r="4"/><circle cx="235" cy="55" r="6"/>
<circle cx="110" cy="12" r="3"/>
</g></svg></div>"""

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)

SKILLS = [
    "python", "sql", "excel", "power bi", "tableau", "pandas",
    "data visualization", "machine learning", "deep learning", "nlp",
    "scikit-learn", "tensorflow", "pytorch", "statistics",
    "data analysis", "javascript", "react", "html", "css", "django",
    "docker", "postgresql", "rest apis", "seo", "google ads",
    "quickbooks", "auditing", "financial reporting", "figma",
]

DEFAULT_JOB = (
    "We are hiring a Data Analyst. Requirements: Python, SQL, pandas, "
    "Excel, dashboards and data visualization. Basic machine learning "
    "and statistics are a plus."
)

GRADS = [
    ("#14b8a6", "#0ea5e9"), ("#8b5cf6", "#ec4899"), ("#f59e0b", "#ef4444"),
    ("#22c55e", "#14b8a6"), ("#3b82f6", "#8b5cf6"), ("#ec4899", "#f97316"),
]
MEDALS = ["🥇", "🥈", "🥉"]


def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)


def find_skills(text, vocabulary):
    text = text.lower()
    found = set()
    for skill in vocabulary:
        pattern = r"(?<![a-z])" + re.escape(skill) + r"(?![a-z])"
        if re.search(pattern, text):
            found.add(skill)
    return found


def find_years(text):
    m = re.search(r"experience:\s*(\d+)", text, re.I)
    if not m:
        m = re.search(r"(\d+)\+?\s*years", text, re.I)
    return int(m.group(1)) if m else 0


def find_summary(text):
    m = re.search(r"summary:\s*(.+)", text, re.I)
    if m:
        return m.group(1).strip()
    return " ".join(text.split())[:250]


def find_title(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1 and len(lines[1]) <= 40 and ":" not in lines[1]:
        return lines[1]
    return ""


def chips(items, kind):
    if not items:
        return '<span class="chip none">none</span>'
    return "".join(
        f'<span class="chip {kind}">{html.escape(i)}</span>' for i in items
    )


def initials(name):
    parts = name.split()
    first = parts[0][0]
    last = parts[-1][0] if len(parts) > 1 else ""
    return (first + last).upper()


def avatar(name, size=64):
    a, b = GRADS[sum(ord(ch) for ch in name) % len(GRADS)]
    return (
        f'<div class="avatar" style="background:linear-gradient(135deg,{a},{b});'
        f'width:{size}px;height:{size}px;font-size:{int(size * 0.38)}px">'
        f'{html.escape(initials(name))}</div>'
    )


def ring(p):
    return (
        '<svg width="92" height="92" viewBox="0 0 36 36">'
        '<circle cx="18" cy="18" r="15.9" fill="none" '
        'stroke="rgba(148,163,184,0.3)" stroke-width="3"/>'
        '<circle cx="18" cy="18" r="15.9" fill="none" stroke="#2dd4bf" '
        f'stroke-width="3" stroke-linecap="round" stroke-dasharray="{p} 100" '
        'transform="rotate(-90 18 18)"/>'
        f'<text x="18" y="21.5" text-anchor="middle" font-size="9" '
        f'font-weight="700" fill="currentColor">{p}%</text></svg>'
    )


# ---------- Sidebar ----------
st.sidebar.header("Settings")
uploaded = st.sidebar.file_uploader(
    "Upload CVs (PDF)", type=["pdf"], accept_multiple_files=True
)
extra_text = st.sidebar.text_input(
    "Extra required skills (comma-separated)", ""
)
min_years = st.sidebar.slider("Minimum years of experience", 0, 10, 0)
skill_weight = st.sidebar.slider(
    "Weight of skills vs. text similarity (%)", 0, 100, 70
)

# ---------- Load CVs ----------
cvs = {}
if uploaded:
    for f in uploaded:
        cvs[os.path.splitext(f.name)[0]] = read_pdf(f)
else:
    st.sidebar.info("No upload yet: using the sample CVs in the cvs folder.")
    for path in sorted(glob.glob("cvs/*.pdf")):
        name = os.path.splitext(os.path.basename(path))[0]
        cvs[name] = read_pdf(path)

job = st.text_area("Job description", DEFAULT_JOB, height=150)

if not cvs:
    st.warning("No CVs found.")
    st.stop()
if not job.strip():
    st.warning("Please enter a job description.")
    st.stop()

# ---------- Scoring ----------
extra = [s.strip().lower() for s in extra_text.split(",") if s.strip()]
vocabulary = sorted(set(SKILLS) | set(extra))
job_skills = find_skills(job, vocabulary) | set(extra)

names = list(cvs.keys())
docs = [job] + [cvs[n] for n in names]
tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
matrix = tfidf.fit_transform(docs)
sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
top_sim = sims.max() if sims.max() > 0 else 1.0

w = skill_weight / 100
rows = []
for name, sim in zip(names, sims):
    text = cvs[name]
    cv_skills = find_skills(text, vocabulary)
    matched = sorted(job_skills & cv_skills)
    missing = sorted(job_skills - cv_skills)
    text_score = sim / top_sim
    if job_skills:
        skill_score = len(matched) / len(job_skills)
        final = w * skill_score + (1 - w) * text_score
    else:
        skill_score = 0.0
        final = text_score
    rows.append({
        "Candidate": name.replace("_", " "),
        "Title": find_title(text),
        "Match": round(final * 100),
        "Skills covered": round(skill_score * 100),
        "Years": find_years(text),
        "Matched skills": ", ".join(matched),
        "Missing skills": ", ".join(missing),
        "Summary": find_summary(text),
    })

df = pd.DataFrame(rows)
df = df[df["Years"] >= min_years]
df = df.sort_values("Match", ascending=False).reset_index(drop=True)

if df.empty:
    st.warning("No candidates meet the minimum experience. "
               "Lower the filter in the sidebar.")
    st.stop()

if job_skills:
    st.write("**Skills detected in the job:** " + ", ".join(sorted(job_skills)))
else:
    st.info("No known skills detected in the job description. "
            "Add some in the sidebar to improve the ranking.")

# ---------- Summary ----------
best = df.iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("CVs shown", len(df))
c2.metric("Top match", best["Candidate"])
c3.metric("Top score", f"{best['Match']}%")

# ---------- Top candidates ----------
st.subheader("Top candidates")
top = df.head(3)
cols = st.columns(len(top))
for col, (rank, row) in zip(cols, top.iterrows()):
    first = " first" if rank == 0 else ""
    col.markdown(
        f'<div class="card{first}">'
        f'<div class="medal">{MEDALS[rank]}</div>'
        f'{avatar(row["Candidate"])}'
        f'<div class="cname">{html.escape(row["Candidate"])}</div>'
        f'<div class="ctitle">{html.escape(row["Title"]) or "&nbsp;"}</div>'
        f'{ring(row["Match"])}'
        f'<div class="cmeta">{row["Years"]} yrs &middot; '
        f'{row["Skills covered"]}% skills</div></div>',
        unsafe_allow_html=True,
    )

# ---------- Chart ----------
fig = px.bar(
    df.sort_values("Match"), x="Match", y="Candidate",
    orientation="h", range_x=[0, 110],
)
fig.update_traces(marker_color="#2dd4bf", texttemplate="%{x}%",
                  textposition="outside")
fig.update_layout(
    height=max(260, 60 + 40 * len(df)),
    xaxis_title=None, yaxis_title=None,
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=10, b=10),
)
fig.update_xaxes(showgrid=False, visible=False)
st.subheader("Match with the job description")
st.plotly_chart(fig)

# ---------- Table ----------
st.subheader("Ranking")
table = df[["Candidate", "Match", "Skills covered", "Years"]].copy()
table.index = table.index + 1
st.dataframe(
    table,
    column_config={
        "Match": st.column_config.ProgressColumn(
            "Match", min_value=0, max_value=100, format="%d%%"
        ),
        "Skills covered": st.column_config.ProgressColumn(
            "Skills covered", min_value=0, max_value=100, format="%d%%"
        ),
    },
)

# ---------- Candidate details ----------
st.subheader("Candidate details")
for rank, row in df.iterrows():
    badge = MEDALS[rank] if rank < 3 else f"#{rank + 1}"
    title = f"{badge}  {row['Candidate']}  -  {row['Match']}%"
    with st.expander(title):
        matched = [s for s in row["Matched skills"].split(", ") if s]
        missing = [s for s in row["Missing skills"].split(", ") if s]
        subtitle = (
            f"{html.escape(row['Title'])} &middot; " if row["Title"] else ""
        )
        st.markdown(
            f'<div class="exp-head">{avatar(row["Candidate"], 52)}'
            f'<div><div class="cname">{html.escape(row["Candidate"])}</div>'
            f'<div class="ctitle">{subtitle}'
            f'{row["Years"]} years of experience</div></div></div>',
            unsafe_allow_html=True,
        )
        st.write(f"**Summary:** {row['Summary']}")
        st.markdown(
            '<div class="chip-title">Matched skills</div>'
            + chips(matched, "ok")
            + '<div class="chip-title">Missing skills</div>'
            + chips(missing, "miss"),
            unsafe_allow_html=True,
        )

# ---------- Download ----------
export = df.drop(columns=["Summary", "Title"])
export.index = export.index + 1
st.download_button(
    "Download ranking (CSV)",
    export.to_csv().encode("utf-8"),
    "ranking.csv",
    "text/csv",
)

st.markdown(
    '<div class="foot">Built with Python, Streamlit and scikit-learn</div>',
    unsafe_allow_html=True,
)