"""
Topic + Aspect-Sentiment Explorer  (Streamlit demo)

Same methodology as Topic_Modeling_Final.ipynb, made domain-agnostic:
  reviews -> sentences (spaCy) -> MiniLM embeddings -> UMAP -> HDBSCAN -> BERTopic
  -> reduce to K topics -> outlier reassignment -> soft multi-topic assignment
  -> noun-chunk aspects -> DeBERTa ABSA sentiment -> omega-weighted topic aggregation
  -> bar chart / word clouds / mean sentiment / stacked sentiment bars

Run:  streamlit run app.py
"""
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

EMBED_MODEL = "all-MiniLM-L6-v2"
ABSA_MODEL = "yangheng/deberta-v3-base-absa-v1.1"
SEED = 42
GAMMA = 0.5          # weight of aspect-topic cosine vs. topic distribution (as in notebook)
MULTI_THRESH = 0.30  # soft multi-topic threshold (as in notebook)
MIN_SENTENCES = 30

SAMPLE = "\n".join([
    "The room was spotless and the bed was very comfortable. The view from the balcony was stunning.",
    "Breakfast buffet had great variety. The coffee was awful though.",
    "Staff at the front desk were friendly and helpful. Check-in took only five minutes.",
    "The room smelled of smoke and the carpet was stained. Housekeeping never came to fix it.",
    "Food at the restaurant was delicious and fresh. The pasta was the best I have had.",
    "Way too expensive for what you get. The price of drinks at the bar is ridiculous.",
    "Service was painfully slow and the waiter was rude. We waited an hour for our meal.",
    "Great location near the metro and shops. Walking to the old town takes ten minutes.",
    "The wifi kept dropping and the air conditioning was loud. The shower had weak water pressure.",
    "Good value for money and the portions were generous. The dessert was overpriced but tasty.",
    "Reception staff solved our booking problem quickly. They were polite and professional.",
    "The pool area was clean and quiet. Sun loungers were always available.",
    "Dinner was cold and tasteless. The menu had very few vegetarian options.",
    "Hidden fees at checkout made the bill much higher. I felt cheated on the final price.",
    "Perfect location for sightseeing and public transport. The street was noisy at night though.",
    "The bathroom was dirty and the towels were old. Management did not care about our complaint.",
    "Lovely staff who remembered our names. They arranged a birthday cake for my wife.",
    "Fresh fruit and excellent omelettes at breakfast. The orange juice was watery.",
    "Beds were hard and the walls were thin. We heard the neighbours all night.",
    "Affordable rates and a decent room. Nothing fancy but good value.",
    "Taxi from the airport was easy to arrange at the desk. The driver was punctual.",
    "The restaurant staff were attentive and kind. Our food arrived quickly and hot.",
    "Parking was expensive and hard to find. The garage entrance is very narrow.",
    "Rooms are modern and bright with a great view. The elevator was slow and often broken.",
])


# ---------------------------------------------------------------------------
# Cached model loaders
# ---------------------------------------------------------------------------
def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner="Loading spaCy model...")
def load_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm", disable=["ner"])
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm", disable=["ner"])


@st.cache_resource(show_spinner="Loading sentence embedding model...")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device=_device())


@st.cache_resource(show_spinner="Loading aspect-sentiment model (first run downloads ~700MB)...")
def load_absa():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ABSA_MODEL, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(ABSA_MODEL).to(_device()).eval()
    return tok, model


# ---------------------------------------------------------------------------
# Step 1: reviews -> sentences (+ noun-chunk aspects)
# ---------------------------------------------------------------------------
def get_aspects(sent, max_k=3):
    def valid(t):
        return len(t) >= 3 and t.isalpha()

    chunks = [c.text.lower().strip() for c in sent.noun_chunks
              if c.root.pos_ in ("NOUN", "PROPN") and valid(c.root.text)]
    if chunks:
        return list(dict.fromkeys(chunks))[:max_k]
    nouns = [t.text.lower() for t in sent
             if t.pos_ in ("NOUN", "PROPN") and not t.is_stop and valid(t.text)]
    return list(dict.fromkeys(nouns))[:max_k]


def split_sentences(texts, nlp):
    rows = []
    for rid, doc in enumerate(nlp.pipe(texts, batch_size=64)):
        for sent in doc.sents:
            s = sent.text.strip()
            if len(s) < 10 or not re.search(r"[A-Za-z0-9]", s):
                continue
            rows.append({"review_id": rid, "sentence": s, "aspects": get_aspects(sent)})
    df = pd.DataFrame(rows)
    df["sentence_id"] = df.index
    return df


# ---------------------------------------------------------------------------
# Step 2: BERTopic with parameters that scale to the corpus size
# ---------------------------------------------------------------------------
def enforce_topics(topics, embeddings, k_target, notes):
    """Return topic ids with no outlier (-1) and at most k_target topics, renumbered 0..K-1 by size."""
    topics = np.array(topics)
    emb = np.asarray(embeddings, dtype=float)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def centroids(ids):
        c = np.vstack([emb[topics == t].mean(axis=0) for t in ids])
        return c / np.linalg.norm(c, axis=1, keepdims=True)

    # 1) outliers -> nearest topic centroid
    out = topics == -1
    if out.any():
        ids = sorted(set(topics) - {-1})
        topics[out] = np.array(ids)[(emb[out] @ centroids(ids).T).argmax(axis=1)]
        notes.append(f"Reassigned {int(out.sum())} outlier sentences to their nearest topic.")

    # 2) too many topics -> merge the smallest into its nearest neighbour
    ids, merged = sorted(set(topics)), 0
    while len(ids) > k_target:
        sizes = {t: int((topics == t).sum()) for t in ids}
        small = min(sizes, key=sizes.get)
        rest = [t for t in ids if t != small]
        topics[topics == small] = rest[int((centroids([small]) @ centroids(rest).T).argmax())]
        ids, merged = rest, merged + 1
    if merged:
        notes.append(f"Merged {merged} small topic(s) to reach the target of {k_target} topics.")

    # 3) renumber 0..K-1, largest topic first
    order = sorted(set(topics), key=lambda t: -int((topics == t).sum()))
    remap = {t: i for i, t in enumerate(order)}
    return [remap[t] for t in topics]


def build_topic_model(sentences, embeddings, embedder, k_target, extra_stopwords, nlp):
    from bertopic import BERTopic
    from bertopic.representation import BaseRepresentation
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
    from umap import UMAP

    n = len(sentences)
    umap_model = UMAP(n_neighbors=min(15, n - 1), n_components=min(5, n - 2),
                      min_dist=0.0, metric="cosine", random_state=SEED)
    mcs = max(5, min(200, round(n * 0.01)))
    hdb = HDBSCAN(min_cluster_size=mcs, min_samples=min(10, mcs), metric="euclidean",
                  cluster_selection_method="eom", prediction_data=True)
    min_df = 1 if n < 300 else (2 if n < 3000 else 5)
    vectorizer = CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS) + extra_stopwords, ngram_range=(1, 2),
        min_df=min_df, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

    class LemmaDedup(BaseRepresentation):
        """Merge morphological variants in each topic's keyword list (plural/verb forms)."""
        def __init__(self, top_n_words=50):
            self.top_n_words, self.cache = top_n_words, {}

        def _lemma(self, phrase):
            if phrase not in self.cache:
                lem = " ".join(t.lemma_.lower() for t in nlp(phrase) if t.lemma_.strip())
                self.cache[phrase] = lem or phrase.lower()
            return self.cache[phrase]

        def extract_topics(self, topic_model, documents, c_tf_idf, topics):
            out = {}
            for tid, pairs in topics.items():
                seen = {}
                for w, s in pairs:
                    seen.setdefault(self._lemma(w), (w, s))
                out[tid] = sorted(seen.values(), key=lambda x: x[1], reverse=True)[: self.top_n_words]
            return out

    def make(cluster_model):
        return BERTopic(embedding_model=embedder, umap_model=umap_model, hdbscan_model=cluster_model,
                        vectorizer_model=vectorizer, ctfidf_model=ctfidf, verbose=False)

    notes = []
    tm = make(hdb)
    tm.fit_transform(sentences, embeddings)
    topics = list(tm.topics_)
    n_found = len(set(topics) - {-1})

    if n_found < 2:  # corpus too small / homogeneous for density clustering
        k_km = int(min(k_target, max(2, n // 15)))
        tm = make(KMeans(n_clusters=k_km, random_state=SEED, n_init=10))
        tm.fit_transform(sentences, embeddings)
        topics = list(tm.topics_)
        notes.append(f"HDBSCAN found <2 clusters on this corpus; used KMeans (k={k_km}) instead.")
    elif n_found > k_target:
        tm.reduce_topics(sentences, nr_topics=k_target + (1 if -1 in tm.topics_ else 0))
        topics = list(tm.topics_)

    # Guarantee a clean result whatever the BERTopic version does with nr_topics:
    # no outlier topic (-1) and at most k_target topics, numbered 0..K-1.
    topics = enforce_topics(topics, embeddings, k_target, notes)

    tm.update_topics(sentences, topics=topics, vectorizer_model=vectorizer, ctfidf_model=ctfidf,
                     representation_model=LemmaDedup(), top_n_words=50)
    return tm, list(topics), notes


def label_topics(tm, topic_ids):
    labels = {}
    for tid in topic_ids:
        words = [w for w, _ in tm.get_topic(tid)][:3]
        labels[tid] = f"T{tid} · " + " · ".join(words)
    return labels


# ---------------------------------------------------------------------------
# Step 3: soft multi-topic assignment -> (sentence, topic, aspect) rows
# ---------------------------------------------------------------------------
def build_absa_frame(sent_df, topics, tm, topic_ids, embeddings, embedder, labels):
    try:
        distr, _ = tm.approximate_distribution(sent_df["sentence"].tolist(), calculate_tokens=False)
        if distr.shape[1] != len(topic_ids):
            distr = None
    except Exception:
        distr = None
    col = {t: j for j, t in enumerate(topic_ids)}

    rows = []
    for i, (sid, rid, sent, aspects) in enumerate(
            zip(sent_df["sentence_id"], sent_df["review_id"], sent_df["sentence"], sent_df["aspects"])):
        if not aspects:
            continue
        hard = topics[i]
        assigned = [(hard, float(distr[i][col[hard]]) if distr is not None else 1.0)]
        if distr is not None:
            d = distr[i]
            strong = [j for j in np.argsort(d)[::-1][:2] if d[j] >= MULTI_THRESH]
            if len(strong) >= 2:
                assigned = [(topic_ids[j], float(d[j])) for j in strong]
        for tid, p in assigned:
            for a in aspects:
                rows.append((sid, rid, sent, tid, p, a))
    df = pd.DataFrame(rows, columns=["sentence_id", "review_id", "sentence", "topic_id",
                                     "topic_distribution", "aspect_keyword"])
    if df.empty:
        return df

    # Aspect-topic cosine similarity against topic centroids (of the final assignment)
    emb = np.asarray(embeddings)
    centroids = np.vstack([emb[np.array(topics) == t].mean(axis=0) for t in topic_ids])
    uniq = df["aspect_keyword"].unique()
    a_emb = embedder.encode(list(uniq), batch_size=128, show_progress_bar=False)
    a_emb = a_emb / np.linalg.norm(a_emb, axis=1, keepdims=True)
    c_emb = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    sim = a_emb @ c_emb.T
    a_idx = df["aspect_keyword"].map({a: i for i, a in enumerate(uniq)}).to_numpy()
    t_idx = df["topic_id"].map(col).to_numpy()
    df["aspect_topic_cosine"] = sim[a_idx, t_idx]

    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    df["aspect_topic_cosine_norm"] = df.groupby(["sentence_id", "aspect_keyword"])["aspect_topic_cosine"].transform(softmax)
    df["topic_label"] = df["topic_id"].map(labels)
    return df


# ---------------------------------------------------------------------------
# Step 4: aspect-based sentiment (DeBERTa ABSA) on unique (sentence, aspect) pairs
# ---------------------------------------------------------------------------
def run_absa(df, progress_cb=None, batch_size=32):
    import torch
    tok, model = load_absa()
    dev = next(model.parameters()).device
    pairs = df.drop_duplicates(["sentence_id", "aspect_keyword"])[["sentence_id", "aspect_keyword", "sentence"]].reset_index(drop=True)
    order = np.argsort(pairs["sentence"].str.len().to_numpy())  # length-sort -> less padding
    probs = np.zeros((len(pairs), model.config.num_labels), dtype=np.float32)

    with torch.no_grad():
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            enc = tok(pairs["sentence"].iloc[idx].tolist(), pairs["aspect_keyword"].iloc[idx].tolist(),
                      truncation=True, padding=True, max_length=128, return_tensors="pt").to(dev)
            probs[idx] = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            if progress_cb:
                progress_cb(min(1.0, (b + batch_size) / len(order)))

    cols = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    for i, c in enumerate(cols):
        pairs[f"prob_{c}"] = probs[:, i]
    out = df.merge(pairs.drop(columns="sentence"), on=["sentence_id", "aspect_keyword"], how="left")
    out["sentiment"] = np.array(cols)[out[[f"prob_{c}" for c in cols]].to_numpy().argmax(axis=1)]
    out["sentiment_score"] = out["prob_positive"] - out["prob_negative"]
    return out


# ---------------------------------------------------------------------------
# Step 5: omega weighting + per-topic aggregation (same formulas as notebook)
# ---------------------------------------------------------------------------
def aggregate(df):
    df = df.copy()
    df["omega_raw"] = GAMMA * df["aspect_topic_cosine_norm"] + (1 - GAMMA) * df["topic_distribution"]
    grp = df.groupby(["sentence_id", "aspect_keyword"])["omega_raw"]
    df["omega"] = grp.transform(lambda x: x / x.sum() if x.sum() > 0 else 1.0 / len(x))
    df["weighted_contribution"] = df["omega"] * df["sentiment_score"]
    for s in ("positive", "neutral", "negative"):
        df[f"w_{s}"] = df["omega"] * (df["sentiment"] == s)
    g = df.groupby("topic_label")
    tot = g["omega"].sum()
    agg = pd.DataFrame({
        "weighted_mean_score": g["weighted_contribution"].sum() / tot,
        "total_weight": tot,
        "pct_positive": 100 * g["w_positive"].sum() / tot,
        "pct_neutral": 100 * g["w_neutral"].sum() / tot,
        "pct_negative": 100 * g["w_negative"].sum() / tot,
    }).sort_values("weighted_mean_score")
    return df, agg


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(texts, k_target, max_sentences, extra_stopwords):
    nlp, embedder = load_spacy(), load_embedder()

    st.write("1/5 Splitting reviews into sentences...")
    texts = [re.sub(r"\s+", " ", str(t)).strip() for t in texts]
    texts = list(dict.fromkeys(t for t in texts if t))  # drop blanks + exact duplicates
    sent_df = split_sentences(texts, nlp)
    notes = []
    if len(sent_df) > max_sentences:
        sent_df = sent_df.sample(max_sentences, random_state=SEED).sort_values("sentence_id").reset_index(drop=True)
        sent_df["sentence_id"] = sent_df.index
        notes.append(f"Corpus capped to a random sample of {max_sentences} sentences for demo speed.")
    if len(sent_df) < MIN_SENTENCES:
        raise ValueError(f"Only {len(sent_df)} usable sentences found; need at least {MIN_SENTENCES}.")

    sentences = sent_df["sentence"].tolist()
    st.write(f"2/5 Embedding {len(sentences)} sentences...")
    embeddings = embedder.encode(sentences, batch_size=64, show_progress_bar=False, convert_to_numpy=True)

    st.write("3/5 Discovering topics (UMAP + HDBSCAN + BERTopic)...")
    tm, topics, tnotes = build_topic_model(sentences, embeddings, embedder, k_target, extra_stopwords, nlp)
    notes += tnotes
    topic_ids = sorted(set(topics))
    labels = label_topics(tm, topic_ids)
    sent_df["topic_id"] = topics
    sent_df["topic_label"] = sent_df["topic_id"].map(labels)

    st.write("4/5 Extracting aspects and scoring aspect-level sentiment...")
    absa = build_absa_frame(sent_df, topics, tm, topic_ids, embeddings, embedder, labels)
    if absa.empty:
        raise ValueError("No aspect terms (nouns) could be extracted from this text.")
    bar = st.progress(0.0)
    absa = run_absa(absa, progress_cb=bar.progress)

    st.write("5/5 Aggregating per topic...")
    absa, agg = aggregate(absa)

    keywords = {t: tm.get_topic(t)[:100] for t in topic_ids}
    return {"absa": absa, "agg": agg, "sent_df": sent_df, "labels": labels, "keywords": keywords,
            "n_reviews": len(texts), "n_sentences": len(sentences), "notes": notes}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def fig_prevalence(res):
    counts = res["sent_df"]["topic_label"].value_counts().rename_axis("topic").reset_index(name="sentences")
    counts["top keywords"] = counts["topic"].map(
        {res["labels"][t]: ", ".join(w for w, _ in res["keywords"][t][:8]) for t in res["labels"]})
    fig = px.bar(counts.sort_values("sentences"), x="sentences", y="topic", orientation="h",
                 hover_data=["top keywords"], title="Topic prevalence (sentences per topic)")
    fig.update_layout(yaxis_title=None, height=60 + 38 * len(counts))
    return fig


def fig_mean_sentiment(agg):
    d = agg.reset_index()
    fig = px.bar(d, x="weighted_mean_score", y="topic_label", orientation="h",
                 color="weighted_mean_score", color_continuous_scale="RdYlGn", range_color=[-1, 1],
                 text=d["weighted_mean_score"].round(2), title="Mean sentiment by topic (-1 negative … +1 positive)")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(yaxis_title=None, xaxis_title="Weighted mean sentiment score",
                      coloraxis_showscale=False, height=60 + 38 * len(d))
    return fig


def fig_stacked(agg):
    d = agg.reset_index()
    fig = go.Figure()
    for col, name, color in [("pct_negative", "negative", "#E24B4A"),
                             ("pct_neutral", "neutral", "#B4B2A9"),
                             ("pct_positive", "positive", "#639922")]:
        fig.add_bar(y=d["topic_label"], x=d[col], name=name, orientation="h", marker_color=color,
                    text=[f"{v:.1f}%" if v > 4 else "" for v in d[col]], textposition="inside")
    fig.update_layout(barmode="stack", title="Sentiment distribution by topic (%)",
                      xaxis=dict(range=[0, 100], title="%"), yaxis_title=None,
                      legend=dict(orientation="h", y=-0.15), height=100 + 38 * len(d))
    return fig


def wordcloud_image(freqs):
    from wordcloud import WordCloud
    return WordCloud(background_color="white", width=800, height=500, max_words=100,
                     colormap="viridis").generate_from_frequencies(dict(freqs)).to_array()


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------
def load_table(upload):
    if upload.name.lower().endswith(".csv"):
        try:
            return pd.read_csv(upload)
        except UnicodeDecodeError:
            upload.seek(0)
            return pd.read_csv(upload, encoding="latin-1")
    return pd.read_excel(upload)


def guess_text_column(df):
    obj = [c for c in df.columns if df[c].dtype == object]
    if not obj:
        return df.columns[0]
    return max(obj, key=lambda c: df[c].astype(str).str.len().mean())


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Topic Modeling Based Sentiment Analysis", layout="wide")
    st.title("Topic Modeling Based Sentiment Analysis")
    st.caption("Paste text or upload a CSV/Excel of reviews, posts or comments (English). "
               "Get topics, per-topic sentiment, word clouds and charts.")

    sb = st.sidebar
    sb.header("1. Input")
    mode = sb.radio("Source", ["Paste text", "Upload CSV / Excel"], horizontal=True)
    texts = []
    if mode == "Paste text":
        sb.button("Load sample reviews", on_click=lambda: st.session_state.update(pasted=SAMPLE))
        sb.text_area("One review/comment per line", key="pasted", height=220)
        texts = [t for t in st.session_state.get("pasted", "").splitlines() if t.strip()]
    else:
        up = sb.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
        if up is not None:
            table = load_table(up)
            col = sb.selectbox("Text column", list(table.columns),
                               index=list(table.columns).index(guess_text_column(table)))
            texts = table[col].dropna().astype(str).tolist()
            sb.caption(f"{len(texts)} rows in '{col}'")

    sb.header("2. Settings")
    k = sb.slider("Target number of topics", 3, 20, 8)
    max_sent = sb.select_slider("Max sentences to analyse", [500, 1000, 2000, 3000, 5000], value=2000,
                                help="The sentiment model is slow on CPU; raise this if you have a GPU.")
    extra = sb.text_input("Extra stop words (comma-separated)", placeholder="e.g. hotel name, city name")
    extra_sw = [w.strip().lower() for w in extra.split(",") if w.strip()]

    if sb.button("Analyze", type="primary", disabled=not texts):
        try:
            with st.status("Running pipeline...", expanded=True) as status:
                st.session_state["result"] = run_pipeline(texts, k, max_sent, extra_sw)
                status.update(label="Done", state="complete", expanded=False)
        except Exception as e:
            st.error(str(e))

    res = st.session_state.get("result")
    if not res:
        st.info("Add some text in the sidebar and click **Analyze**.")
        return

    for n in res["notes"]:
        st.caption(f"ℹ️ {n}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews", res["n_reviews"])
    c2.metric("Sentences", res["n_sentences"])
    c3.metric("Topics", len(res["labels"]))

    t_over, t_cloud, t_ex, t_data = st.tabs(["Overview", "Word clouds", "Example sentences", "Data"])

    with t_over:
        st.plotly_chart(fig_prevalence(res), use_container_width=True)
        left, right = st.columns(2)
        left.plotly_chart(fig_mean_sentiment(res["agg"]), use_container_width=True)
        right.plotly_chart(fig_stacked(res["agg"]), use_container_width=True)

    with t_cloud:
        ids = list(res["labels"])
        for r in range(0, len(ids), 3):
            cols = st.columns(3)
            for c, tid in zip(cols, ids[r:r + 3]):
                c.image(wordcloud_image(res["keywords"][tid]), caption=res["labels"][tid])

    with t_ex:
        topic = st.selectbox("Topic", list(res["agg"].index[::-1]))
        d = res["absa"][res["absa"]["topic_label"] == topic].drop_duplicates(["sentence_id", "aspect_keyword"])
        d = d[["sentence", "aspect_keyword", "sentiment", "sentiment_score"]].sort_values("sentiment_score")
        a, b = st.columns(2)
        a.subheader("Most negative")
        a.dataframe(d.head(8), hide_index=True, use_container_width=True)
        b.subheader("Most positive")
        b.dataframe(d.tail(8).iloc[::-1], hide_index=True, use_container_width=True)

    with t_data:
        st.dataframe(res["agg"].round(3), use_container_width=True)
        st.download_button("Download topic summary (CSV)", res["agg"].round(4).to_csv().encode(),
                           "topic_summary.csv", "text/csv")
        st.download_button("Download full results (CSV)", res["absa"].to_csv(index=False).encode(),
                           "absa_results.csv", "text/csv")


if __name__ == "__main__":
    main()
