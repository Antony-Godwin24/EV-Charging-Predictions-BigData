import os, io, uuid, shutil, tempfile, textwrap, datetime as dt, re, urllib.parse
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF

from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

WC_AVAILABLE = True
try:
    from wordcloud import WordCloud, STOPWORDS
except Exception:
    WC_AVAILABLE = False

# Safe profanity import with fallback so the app never crashes if the package is missing
try:
    from better_profanity import profanity
except Exception:
    class _NoProf:
        def load_censor_words(self): pass
        def censor(self, t): return t or ""
    profanity = _NoProf()

st.set_page_config(page_title="✈️ Twitter Sentiment", layout="centered")

# ---------------- Theme & Styling ----------------
GRADIENTS_LIGHT = {
    "Candy (pink→peach→sky)": "linear-gradient(180deg, #ff9a9e 0%, #fecfef 45%, #a1c4fd 100%)",
    "Aurora (teal→mint→violet)": "linear-gradient(180deg, #10b981 0%, #99f6e4 45%, #a78bfa 100%)",
    "Sunset (coral→gold→plum)": "linear-gradient(180deg, #fb7185 0%, #fbbf24 45%, #a78bfa 100%)",
    "Lagoon (aqua→sky→indigo)": "linear-gradient(180deg, #34d399 0%, #60a5fa 50%, #6366f1 100%)",
    "Peach Fizz (peach→rose)": "linear-gradient(180deg, #ffd1a6 0%, #ff9eb5 100%)",
    "Citrus Mint (lime→mint)": "linear-gradient(180deg, #bef264 0%, #34d399 100%)",
    "Bubblegum (rose→lavender)": "linear-gradient(180deg, #ffafbd 0%, #ffc3a0 40%, #cbb4d4 100%)",
    "Skyberry (blue→orchid)": "linear-gradient(180deg, #74ebd5 0%, #acb6e5 100%)",
    "Cocoa Cream (tan→rose)": "linear-gradient(180deg, #f6d365 0%, #fda085 100%)",
}
GRADIENTS_DARK = {
    "Cosmos (navy→violet)": "linear-gradient(180deg, #0f172a 0%, #312e81 100%)",
    "Neon (teal→purple)": "linear-gradient(180deg, #0f766e 0%, #7c3aed 100%)",
    "Ember (charcoal→amber)": "linear-gradient(180deg, #111827 0%, #b45309 100%)",
    "Aurora Night (cyan→indigo)": "linear-gradient(180deg, #0891b2 0%, #3730a3 100%)",
    "Midnight Bloom (blue→pink)": "linear-gradient(180deg, #1e3a8a 0%, #db2777 100%)",
    "Forest Dusk (green→slate)": "linear-gradient(180deg, #065f46 0%, #1f2937 100%)",
    "Midnight Teal (teal→black)": "linear-gradient(180deg, #134e4a 0%, #0b0f14 100%)",
    "Plum Fog (plum→slate)": "linear-gradient(180deg, #3b0764 0%, #111827 100%)",
    "Steel Glow (slate→violet)": "linear-gradient(180deg, #111827 0%, #6d28d9 100%)",
}

st.sidebar.header("🎨 Theme & Controls")
appearance = st.sidebar.radio("Appearance", ["Light", "Dark"], index=0)
grad_list = list(GRADIENTS_LIGHT.keys()) if appearance == "Light" else list(GRADIENTS_DARK.keys())
grad_choice = st.sidebar.selectbox("Gradient", grad_list, index=0)
GRADIENT = (GRADIENTS_LIGHT if appearance == "Light" else GRADIENTS_DARK)[grad_choice]

TEXT_COLOR = "#0f172a" if appearance == "Light" else "#f1f5f9"
CARD_BG = "rgba(255,255,255,0.94)" if appearance == "Light" else "rgba(17,24,39,0.80)"
BORDER = "#e2e8f0" if appearance == "Light" else "#334155"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{ background: {GRADIENT}; }}
[data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
.block-container {{
  color:{TEXT_COLOR}; background: {CARD_BG}; padding: 1.2rem 1.2rem 2rem 1.2rem;
  border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.10); border: 1px solid {BORDER};
}}
h1, h2, h3, label, .stMarkdown, .stTextInput, .stDataFrame {{ color:{TEXT_COLOR}; }}
.metric-card {{ background:rgba(248,250,252,0.6); border:1px solid {BORDER}; padding:.75rem 1rem; border-radius:10px; }}
.download-row button {{ background:linear-gradient(90deg,#6366f1,#22c55e)!important; color:white!important; }}
/* Share bar styles */
.sharebar {{ display:flex; gap:.5rem; flex-wrap:wrap; }}
.sharebtn {{
  display:inline-flex; align-items:center; justify-content:center;
  width:40px; height:40px; border-radius:50%; color:white;
  text-decoration:none; box-shadow:0 2px 6px rgba(0,0,0,0.12);
}}
.wa {{ background:#25D366; }}
.tg {{ background:#229ED9; }}
.x  {{ background:#0f1419; }}
.li {{ background:#0A66C2; }}
.em {{ background:#475569; }}
.sharebtn svg {{ width:22px; height:22px; fill:white; }}
</style>
""", unsafe_allow_html=True)  # Allows custom HTML/CSS in Streamlit markdown. [web:423]

sns.set_theme(style="whitegrid", context="talk")

# ---------------- Performance knobs ----------------
CACHE_TTL = 3600  # seconds

# ---------------- Data loading ----------------
RAW = "dataset/twitter_sentiment.csv"
CLEAN = "dataset/twitter_sentiment_clean.csv"

def base_clean_text(txt):
    if pd.isna(txt): return ""
    txt = re.sub(r"http\S+","",str(txt))
    txt = re.sub(r"@[A-Za-z0-9_]+","",txt)
    txt = re.sub(r"#[A-Za-z0-9_]+","",txt)
    txt = re.sub(r"[^a-zA-Z\s]"," ",txt).lower()
    txt = re.sub(r"\s+"," ",txt).strip()
    return txt

def ensure_clean_csv():
    if os.path.exists(CLEAN):
        return pd.read_csv(CLEAN)
    df = pd.read_csv(RAW)
    if "clean_text" not in df.columns:
        df["clean_text"] = df["text"].apply(base_clean_text)
    os.makedirs(os.path.dirname(CLEAN), exist_ok=True)
    df.to_csv(CLEAN, index=False)
    return df

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_clean_df():
    df = ensure_clean_csv()
    for col in ("airline_sentiment","airline","negativereason"):
        if col in df.columns:
            df[col] = df[col].astype("category")
    if "tweet_created" in df.columns:
        df["tweet_created_dt"] = pd.to_datetime(df["tweet_created"], errors="coerce")
    profanity.load_censor_words()
    def remove_profanity(text: str) -> str:
        if not isinstance(text, str): return ""
        censored = profanity.censor(text)
        return re.sub(r"\*+", " ", censored).strip()
    df["display_text"] = df["clean_text"].apply(remove_profanity)
    df["airline_sentiment"] = df["airline_sentiment"].str.lower()
    keep = {"positive","negative","neutral"}
    df = df[df["airline_sentiment"].isin(keep)].copy()
    return df

df = load_clean_df()
sentiments = ["positive","negative","neutral"]

# ---------------- Sidebar controls ----------------
viz_choice = st.sidebar.radio("Visualization", ["Overall Sentiment", "Sentiment by Airline", "Top Negative Reasons"])
top_n_airlines = st.sidebar.slider("Top airlines", 3, 12, 8, 1)
top_n_reasons = st.sidebar.slider("Top negative reasons", 5, 20, 12, 1)
granularity = st.sidebar.radio("Timeline granularity", ["Day","Week","Month"], index=1)

# Airline filter (applies to all visuals)
all_airlines = sorted(pd.Series(df["airline"]).dropna().unique().tolist())
selected_airlines = st.sidebar.multiselect(
    "Filter airlines (charts & tables)",
    options=all_airlines,
    default=[],
    placeholder="All airlines",
    width="stretch",
)  # Multiselect used to drive a filtered dataframe for all visuals. [web:412]

def apply_airline_filter(frame: pd.DataFrame) -> pd.DataFrame:
    if selected_airlines:
        return frame[frame["airline"].isin(selected_airlines)].copy()
    return frame.copy()

df_viz = apply_airline_filter(df)

# ---------------- Share bar with fixed public URL + logo buttons ----------------
APP_PUBLIC_URL = "https://twitter-airline-sentiment-analysis-from-krish-cs.streamlit.app/"
ENC_URL = urllib.parse.quote_plus(APP_PUBLIC_URL)

with st.sidebar.expander("Share this app"):
    share_html = f"""
    <div class="sharebar" title="Share this app">
      <!-- WhatsApp -->
      <a class="sharebtn wa" aria-label="WhatsApp"
         href="https://wa.me/?text={ENC_URL}" target="_blank" rel="noopener">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M20.5 3.5A11 11 0 0 0 2.2 18.3L1 23l4.8-1.2A11 11 0 1 0 20.5 3.5zM12 20.3a8.3 8.3 0 0 1-4.2-1.1l-.3-.2-2.5.6.7-2.4-.2-.3A8.3 8.3 0 1 1 20.3 12 8.3 8.3 0 0 1 12 20.3zm4.6-6.2c-.3-.2-1.7-.9-2-.9s-.5 0-.7.3-.8.9-1 .9-.5 0-.8-.3a6.2 6.2 0 0 1-1.8-1.6c-.4-.5-.7-1-.5-1.3s.2-.3.4-.5l.2-.3c.1-.2 0-.4 0-.5l-.7-1.8c-.2-.5-.4-.5-.6-.5h-.5a1 1 0 0 0-.7.4 2.9 2.9 0 0 0-.9 2.1 5 5 0 0 0 1.1 2.5c.3.4 2.2 3.2 5.4 4.4l1.7.6a4 4 0 0 0 1.8.1 2.4 2.4 0 0 0 1.6-1.1 2 2 0 0 0 .1-1.1c-.1-.1-.2-.2-.4-.3z"/>
        </svg>
      </a>
      <!-- Telegram -->
      <a class="sharebtn tg" aria-label="Telegram"
         href="https://t.me/share/url?url={ENC_URL}" target="_blank" rel="noopener">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M22 2 2.9 9.2c-.8.3-.8.8-.1 1l4.9 1.5 1.9 6c.3.8.7.9 1.1.2l2.7-4 5.2 3.9c.6.4 1 .2 1.1-.6L23.6 3c.2-.9-.3-1.3-1.6-.8z"/>
        </svg>
      </a>
      <!-- X (Twitter) -->
      <a class="sharebtn x" aria-label="Post on X"
         href="https://twitter.com/intent/tweet?url={ENC_URL}&text=Check%20this%20dashboard" target="_blank" rel="noopener">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M18.2 2H21l-6.6 7.5L22 22h-6.8l-5.3-6.9L3.9 22H1l7.1-8.1L2 2h6.9l4.8 6.3L18.2 2zM16.8 20h1.9L7.3 4H5.4l11.4 16z"/>
        </svg>
      </a>
      <!-- LinkedIn -->
      <a class="sharebtn li" aria-label="LinkedIn"
         href="https://www.linkedin.com/sharing/share-offsite/?url={ENC_URL}" target="_blank" rel="noopener">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M4.98 3.5C4.98 4.88 3.86 6 2.5 6S0 4.88 0 3.5 1.12 1 2.5 1s2.48 1.12 2.48 2.5zM.5 8h4V23h-4V8zm7.98 0h3.84v2.05h.06c.53-.99 1.84-2.05 3.79-2.05 4.05 0 4.8 2.66 4.8 6.12V23h-4v-6.6c0-1.57-.03-3.6-2.2-3.6-2.2 0-2.54 1.72-2.54 3.5V23h-3.75V8z"/>
        </svg>
      </a>
      <!-- Email -->
      <a class="sharebtn em" aria-label="Email"
         href="mailto:?subject=Twitter%20Sentiment%20Dashboard&body={ENC_URL}" target="_blank" rel="noopener">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M20 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2zm0 4-8 5-8-5V6l8 5 8-5v2z"/>
        </svg>
      </a>
    </div>
    """
    st.markdown(share_html, unsafe_allow_html=True)  # Custom share buttons rendered via HTML/SVG. [web:423]
    st.caption(f"Public URL: {APP_PUBLIC_URL}")  # Read-only display of the live URL. [web:407]

# ---------------- KPIs ----------------
st.markdown("## ✈️ Twitter Airline Sentiment Dashboard")
c1, c2, c3 = st.columns(3)
with c1: st.markdown(f"<div class='metric-card'>🧾 Total tweets<br><b>{len(df_viz):,}</b></div>", unsafe_allow_html=True)
with c2: st.markdown(f"<div class='metric-card'>😠 Negative share<br><b>{(df_viz['airline_sentiment'].eq('negative').mean()*100):.1f}%</b></div>", unsafe_allow_html=True)
with c3: st.markdown(f"<div class='metric-card'>🏢 Airlines<br><b>{pd.Series(df_viz['airline']).nunique()}</b></div>", unsafe_allow_html=True)

# ---------------- Plot helpers ----------------
def wrap_labels(labels, width=20):
    return [textwrap.fill(str(l), width=width) for l in labels]

def fig_to_buf(fig, dpi=170):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0); plt.close(fig)
    return buf

def show_overall(frame: pd.DataFrame):
    counts = frame["airline_sentiment"].value_counts().reindex(sentiments).fillna(0)
    s = pd.DataFrame({"sentiment": counts.index, "count": counts.values})
    fig, ax = plt.subplots(figsize=(7.5,4.5), constrained_layout=True)
    sns.barplot(data=s, x="sentiment", y="count", hue="sentiment", palette="pastel", legend=False, ax=ax)
    ax.set_xlabel("Sentiment"); ax.set_ylabel("Number of tweets"); ax.set_title("Overall Sentiment Distribution", pad=10)
    ax.bar_label(ax.containers[0], fmt="%d", padding=3)
    return fig_to_buf(fig)

def show_by_airline(frame: pd.DataFrame, n=6):
    top_air = frame["airline"].value_counts().nlargest(n).index
    dfa = frame[frame["airline"].isin(top_air)]
    if dfa.empty:
        fig, ax = plt.subplots(figsize=(8.5,4.8), constrained_layout=True)
        ax.text(0.5, 0.5, "No data for selected airlines", ha="center", va="center")
        return fig_to_buf(fig)
    pivot = dfa.groupby(["airline","airline_sentiment"]).size().unstack(fill_value=0).reindex(columns=sentiments).loc[top_air]
    fig, ax = plt.subplots(figsize=(8.5,4.8), constrained_layout=True)
    pivot.plot(kind="bar", stacked=True, ax=ax, color=sns.color_palette("husl", 3))
    ax.set_xlabel("Airline"); ax.set_ylabel("Tweet count"); ax.set_title("Sentiment by Airline (stacked)", pad=10)
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02,1), loc="upper left")
    ax.set_xticklabels(wrap_labels(pivot.index, 14), rotation=15, ha="right")
    return fig_to_buf(fig)

def show_neg_reasons(frame: pd.DataFrame, k=8):
    neg = frame[frame["airline_sentiment"]=="negative"]
    top = neg["negativereason"].dropna().astype(str).value_counts().nlargest(k)
    if top.empty:
        fig, ax = plt.subplots(figsize=(8.8,5.2), constrained_layout=True)
        ax.text(0.5, 0.5, "No negative reasons in selection", ha="center", va="center")
        return fig_to_buf(fig)
    fig, ax = plt.subplots(figsize=(8.8,5.2), constrained_layout=True)
    sns.barplot(y=wrap_labels(top.index, 24), x=top.values, orient="h", palette=sns.color_palette("crest", n_colors=k), ax=ax)
    ax.set_xlabel("Number of tweets"); ax.set_ylabel("Negative reason"); ax.set_title("Top Negative Reasons", pad=10)
    for c in ax.containers: ax.bar_label(c, fmt="%d", padding=3)
    return fig_to_buf(fig)

# ---------------- Show one chart ----------------
st.markdown("### 📊 Visualization")
if viz_choice == "Overall Sentiment":
    st.image(show_overall(df_viz), width="stretch", caption="Overall distribution")
elif viz_choice == "Sentiment by Airline":
    st.image(show_by_airline(df_viz, top_n_airlines), width="stretch", caption="Stacked counts by airline")
else:
    st.image(show_neg_reasons(df_viz, top_n_reasons), width="stretch", caption="Top drivers of negative tweets")

# ---------------- Analyzer: HashingVectorizer + cosine ----------------
st.markdown("### 📝 Your Tweet Analyzer")

try:
    sia = SentimentIntensityAnalyzer()
except Exception:
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

def predict_vader(txt):
    scores = sia.polarity_scores(txt or "")
    comp = scores.get("compound", 0.0)
    label = "positive" if comp >= 0.05 else "negative" if comp <= -0.05 else "neutral"
    return label, scores

@st.cache_data(ttl=CACHE_TTL, show_spinner=False, max_entries=2)
def build_hash_corpus(series):
    s = pd.Series(series).fillna("").astype(str).str.lower()
    s = s.str.replace(r"[^a-z\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    s = s[s.str.len() >= 2]
    if s.empty:
        return None, None, None
    hv = HashingVectorizer(n_features=2**18, alternate_sign=False, norm="l2", ngram_range=(1,2))
    X = hv.transform(s.tolist())
    return hv, X, s.index

hv, Xh, row_idx = build_hash_corpus(df["display_text"])

user_tweet = st.text_area("Type a tweet to analyze and find similar tweets", placeholder="e.g., My flight got cancelled and support isn’t responding 😡")

if st.button("🔍 Analyze my tweet"):
    qt = re.sub(r"\s+"," ", (user_tweet or "")).strip().lower()
    label, scores = predict_vader(qt)
    st.success(f"Predicted sentiment: {label.upper()} | compound={scores['compound']:.3f}")
    if hv is None or Xh is None or row_idx is None:
        st.warning("Similarity unavailable: no usable corpus after preprocessing.")
    elif len(qt) < 3:
        st.warning("Please enter at least 3 letters for a meaningful match.")
    else:
        qv = hv.transform([qt])
        if qv.nnz:
            sims = cosine_similarity(qv, Xh).ravel()
            order = sims.argsort()[::-1][:10]
            idx = row_idx[order]; simvals = sims[order].tolist()
        else:
            base = df.loc[row_idx, "display_text"].fillna("").tolist()
            first_tok = qt.split()[0] if qt.split() else ""
            cand_mask = [first_tok in t for t in base] if first_tok else [True]*len(base)
            cand = [(i, SequenceMatcher(None, qt, base[i]).ratio()) for i in range(len(base)) if cand_mask[i]]
            cand.sort(key=lambda x: x[1], reverse=True)
            picks = [i for i,_ in cand[:10]]
            idx = row_idx[picks]; simvals = [s for _,s in cand[:10]]
        cols = ["tweet_id","airline","airline_sentiment","negativereason","display_text"]
        res = df.loc[idx, cols].copy().rename(columns={"display_text":"text"})
        res["similarity"] = simvals
        st.dataframe(res.reset_index(drop=True), width="stretch")

# ---------------- Keyword Timeline (fast & with related terms) ----------------
st.markdown("### 📈 Keyword Timeline (with related terms)")

def tokenize_corpus(series):
    return series.str.lower().str.replace(r"[^a-z\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True)

@st.cache_data(ttl=CACHE_TTL, show_spinner=False, max_entries=16)
def related_terms_for_keyword(df_in, keyword, top_k=3):
    norm = df_in["display_text"].fillna("").str.lower()
    q = re.escape(keyword.lower())
    mask = norm.str.contains(rf"\b{q}\b", regex=True)
    idx = df_in.index[mask]
    if len(idx) == 0:
        return []
    vec = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-z]{2,}\b")
    X = vec.fit_transform(norm.loc[idx])
    vocab = vec.get_feature_names_out()
    cts = X.sum(axis=0).A1
    s = pd.Series(cts, index=vocab).sort_values(ascending=False)
    s = s.drop(labels=[keyword.lower()], errors="ignore")
    return s.head(top_k).index.tolist()

def timeline_series(df_in, term, granularity):
    if "tweet_created_dt" not in df_in.columns:
        return pd.Series(dtype=int), None
    rule = {"Day":"D","Week":"W","Month":"M"}[granularity]
    norm = df_in["display_text"].fillna("").str.lower()
    mask = norm.str.contains(rf"\b{re.escape(term)}\b", regex=True)
    sub = df_in.loc[mask].dropna(subset=["tweet_created_dt"])
    if sub.empty:
        return pd.Series(dtype=int), None
    start = sub["tweet_created_dt"].min().floor("D")
    end = sub["tweet_created_dt"].max().ceil("D")
    ts = (sub.set_index("tweet_created_dt").sort_index().resample(rule).size())
    full = pd.date_range(start, end, freq=rule)
    ts = ts.reindex(full, fill_value=0)
    return ts, (start, end)

kw = st.text_input("Keyword (we will chart it by default and add related terms if present)")
frame_for_timeline = df_viz
if kw:
    ts_main, span = timeline_series(frame_for_timeline, kw, granularity)
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(figsize=(9.5,4.0), constrained_layout=True)
    if ts_main.empty:
        ax.plot([pd.Timestamp.today().floor('D')], [0], marker="o", color="#0ea5e9", label=kw.lower())
        ax.set_title(f"No dated tweets for '{kw}'. Showing placeholder.")
    else:
        ax.plot(ts_main.index, ts_main.values, marker="o", linewidth=2.2, label=kw.lower())
        rel = related_terms_for_keyword(frame_for_timeline, kw, top_k=3)
        for t in rel:
            ts_t, _ = timeline_series(frame_for_timeline, t, granularity)
            if not ts_t.empty:
                ax.plot(ts_t.index, ts_t.values, marker="o", alpha=0.9, label=t)
        ax.set_title(f"Keyword timeline for '{kw}' and related terms")
    ax.set_xlabel("Date"); ax.set_ylabel(f"Mentions per {granularity.lower()}");
    loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.tick_params(axis="x", labelrotation=30)
    ax.legend(ncol=2, fontsize=9, frameon=False)
    st.image(fig_to_buf(fig), width="stretch")
else:
    st.caption("Enter a keyword to render its timeline quickly (always shows a default line for the entered word).")

# ---------------- Word Cloud (cached text) ----------------
st.markdown("### ☁️ Word Cloud")
sel_wc_sent = st.selectbox("Pick sentiment for cloud", sentiments)

@st.cache_data(ttl=CACHE_TTL, show_spinner=False, max_entries=6)
def wc_text_for_sentiment(df_in, label):
    return " ".join(df_in[df_in["airline_sentiment"]==label]["display_text"].dropna().tolist()[:60000])

if WC_AVAILABLE:
    text_wc = wc_text_for_sentiment(df_viz, sel_wc_sent)
    stop = set(STOPWORDS) | {"rt","https","http"}
    wc = WordCloud(width=900, height=350, background_color="white", colormap="magma", stopwords=stop).generate(text_wc)
    fig, ax = plt.subplots(figsize=(9,3.8), constrained_layout=True); ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    st.image(fig_to_buf(fig), width="stretch")
else:
    st.warning("wordcloud not installed. Run: pip install wordcloud")

# ---------------- PDF (Unicode font with fallback sanitizer) ----------------
FONT_REG = "fonts/DejaVuSans.ttf"
FONT_BOLD = "fonts/DejaVuSans-Bold.ttf"

def safe_pdf_text(s: str) -> str:
    s = (s or "").replace("\u2011","-").replace("\u2013","-").replace("\u2014","-")
    return s.encode("latin-1","replace").decode("latin-1")

def temp_png(buf):
    td = tempfile.mkdtemp(prefix="sent_")
    path = os.path.join(td, f"{uuid.uuid4().hex}.png")
    with open(path, "wb") as f: f.write(buf.getvalue())
    return td, path

def build_pdf():
    b1, b2, b3 = show_overall(df_viz), show_by_airline(df_viz, top_n_airlines), show_neg_reasons(df_viz, top_n_reasons)
    tmp_dirs, paths = [], []
    for b in (b1,b2,b3):
        d, p = temp_png(b); tmp_dirs.append(d); paths.append(p)

    pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=15)
    unicode_ready = os.path.exists(FONT_REG)
    if unicode_ready:
        pdf.add_font("dejavu", "", FONT_REG)
        if os.path.exists(FONT_BOLD): pdf.add_font("dejavu", "B", FONT_BOLD)

    def set_font(bold=False, size=12):
        if unicode_ready: pdf.set_font("dejavu", "B" if bold else "", size)
        else: pdf.set_font("Arial", "B" if bold else "", size)

    pdf.add_page()
    set_font(True, 20)
    title = "Twitter Airline Sentiment Report"
    pdf.cell(0, 12, title if unicode_ready else safe_pdf_text(title), ln=True, align="C")
    pdf.ln(6)
    set_font(False, 12)
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"Generated: {now}"
    pdf.cell(0, 8, gen if unicode_ready else safe_pdf_text(gen), ln=True, align="C")
    pdf.ln(8)
    kpis = [f"Total tweets: {len(df_viz):,}",
            f"Negative share: {(df_viz['airline_sentiment'].eq('negative').mean()*100):.1f}%",
            f"Airlines: {pd.Series(df_viz['airline']).nunique()}"]
    for k in kpis:
        pdf.cell(0, 8, k if unicode_ready else safe_pdf_text(k), ln=True, align="C")
    pdf.ln(6)

    sections = [
        ("Overall Sentiment",
         "Distribution of positive, negative, and neutral opinions across all airlines.",
         paths[0], "Figure 1. Overall sentiment"),
        ("Per-Airline Sentiment",
         "Stacked counts across the most-mentioned airlines to highlight relative differences.",
         paths[1], "Figure 2. Sentiment by airline"),
        ("Drivers of Negative Sentiment",
         "Top-frequency reasons for negative tweets, including cancellations, delays, and service issues.",
         paths[2], "Figure 3. Negative reasons"),
    ]
    for title, body, img, cap in sections:
        pdf.add_page()
        set_font(True, 14); pdf.cell(0, 10, title if unicode_ready else safe_pdf_text(title), ln=True)
        set_font(False, 12)
        pdf.multi_cell(0, 7, body if unicode_ready else safe_pdf_text(body)); pdf.ln(2)
        pdf.image(img, w=180); pdf.ln(2)
        set_font(False, 10)
        pdf.cell(0, 6, cap if unicode_ready else safe_pdf_text(cap), ln=True, align="C")

    data = pdf.output(dest="S").encode("latin-1", errors="ignore")
    for d in tmp_dirs:
        try: shutil.rmtree(d)
        except Exception: pass
    return io.BytesIO(data)

st.markdown("---")
st.subheader("📄 Report")
pdf_buf = build_pdf()
st.download_button("Download PDF report", data=pdf_buf, file_name="twitter_sentiment_report.pdf", mime="application/pdf")
