"""
Topic Modeling Based Sentiment Analysis  (Streamlit demo)

Methodology of Topic_Modeling_Final.ipynb, made domain-agnostic and fully automatic:
  reviews -> sentences (spaCy) -> MiniLM embeddings -> UMAP (once)
  -> HDBSCAN sweep scaled to corpus size
  -> number of topics chosen by max(C_v x TD)  [Strategy A: natural HDBSCAN k,
                                                 Strategy B: c-TF-IDF reduction of a base model]
  -> outlier reassignment -> soft multi-topic assignment
  -> single-word opinion-bearing aspects -> DeBERTa ABSA sentiment
  -> omega-weighted topic aggregation -> charts

The only user inputs are the data and (optionally) stop words.

Run:  streamlit run app.py
"""
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

EMBED_MODEL = "all-MiniLM-L6-v2"
ABSA_MODEL = "yangheng/deberta-v3-base-absa-v1.1"
SEED = 42
GAMMA = 0.5          # weight of aspect-topic cosine vs. topic distribution (as in notebook)
MULTI_THRESH = 0.30  # soft multi-topic threshold (as in notebook)
MIN_SENTENCES = 30
MAX_SENTENCES = 3000  # demo cap (the sentiment model is slow on CPU); raise on a GPU / bigger server
MAX_CANDIDATES_A = 6  # cap on Strategy-A candidates evaluated with C_v x TD
MAX_CANDIDATES_B = 8  # cap on Strategy-B candidates evaluated with C_v x TD

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
# Step 1: reviews -> sentences + single-word, opinion-bearing aspects
# ---------------------------------------------------------------------------
GENERIC_ASPECTS = {
    "thing", "things", "time", "times", "day", "days", "way", "lot", "bit", "kind", "sort", "one", "ones",
    "stuff", "something", "everything", "anything", "nothing", "people", "person", "guy", "guys", "fact",
    "reason", "point", "part", "couple", "number", "type", "year", "years", "week", "weeks", "month",
}
OPINION_VERBS = {"love", "like", "hate", "enjoy", "recommend", "adore", "dislike", "appreciate",
                 "regret", "avoid", "praise", "admire", "prefer", "thank", "dread", "miss"}


def _is_aspect_noun(t, blocked):
    lemma = t.lemma_.lower()
    return (t.pos_ in ("NOUN", "PROPN") and t.is_alpha and len(t.text) >= 3 and not t.is_stop
            and lemma not in GENERIC_ASPECTS and lemma not in blocked and t.text.lower() not in blocked)


def _carries_opinion(t):
    """True if an opinion word is attached to this noun: 'clean room', 'the room was dirty', 'I loved the food'."""
    base = t
    while base.dep_ == "conj" and base.head is not base:  # 'the room and the bed were comfortable'
        base = base.head
    if any(c.dep_ == "amod" and c.pos_ == "ADJ" for c in t.children):
        return True
    head = base.head
    if base.dep_ in ("nsubj", "nsubjpass"):
        if head.pos_ == "ADJ":
            return True
        if any(c.pos_ == "ADJ" and c.dep_ in ("acomp", "attr", "oprd", "xcomp", "advmod") for c in head.children):
            return True
    if base.dep_ in ("dobj", "obj") and head.pos_ == "VERB" and head.lemma_.lower() in OPINION_VERBS:
        return True
    return False


def get_aspects(sent, blocked=frozenset(), require_opinion=True, max_k=3):
    """Single-word aspect terms (noun lemmas); by default only nouns that an opinion word is attached to."""
    out = [t.lemma_.lower() for t in sent
           if _is_aspect_noun(t, blocked) and (not require_opinion or _carries_opinion(t))]
    return list(dict.fromkeys(out))[:max_k]


def split_sentences(texts, nlp, blocked):
    rows = []
    for rid, doc in enumerate(nlp.pipe(texts, batch_size=64)):
        for sent in doc.sents:
            s = sent.text.strip()
            if len(s) < 10 or not re.search(r"[A-Za-z0-9]", s):
                continue
            rows.append({"review_id": rid, "sentence": s,
                         "aspects": get_aspects(sent, blocked),
                         "aspects_any": get_aspects(sent, blocked, require_opinion=False)})
    df = pd.DataFrame(rows)
    df["sentence_id"] = df.index
    return df


# ---------------------------------------------------------------------------
# Step 2: adaptive topic modelling (scaled to corpus size, K by C_v x TD)
# ---------------------------------------------------------------------------
class PrecomputedUMAP:
    """Stand-in for UMAP so the (slow) reduction is done once and reused by every candidate model."""

    def __init__(self, reduced):
        self.reduced = reduced

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.reduced

    def fit_transform(self, X, y=None):
        return self.reduced


def reduce_embeddings(embeddings):
    from umap import UMAP
    n = len(embeddings)
    return UMAP(n_neighbors=min(15, n - 1), n_components=min(5, n - 2), min_dist=0.0,
                metric="cosine", random_state=SEED).fit_transform(embeddings)


def adaptive_settings(n):
    """Search ranges that depend on the amount of data (the notebook's fixed grid only suits large corpora)."""
    k_max = int(np.clip(round(np.sqrt(n) / 3.5), 4, 25))
    k_min = 2 if n < 120 else 3
    floor = 3 if n < 100 else 5
    fracs = [0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.10]
    mcs_list = sorted({int(np.clip(round(f * n), floor, 400)) for f in fracs})
    return k_min, k_max, mcs_list


def ms_options(mcs):
    return sorted({m for m in (max(2, min(mcs, 10)), max(2, mcs // 4), max(2, mcs // 2)) if m <= mcs})


def thin(items, m):
    items = list(items)
    if len(items) <= m:
        return items
    idx = sorted(set(np.linspace(0, len(items) - 1, m).round().astype(int)))
    return [items[i] for i in idx]


def sweep_hdbscan(reduced, mcs_list, method="eom"):
    from hdbscan import HDBSCAN
    X = np.asarray(reduced, dtype=np.float64)
    rows = []
    for mcs in mcs_list:
        for ms in ms_options(mcs):
            labels = HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean",
                             cluster_selection_method=method).fit_predict(X)
            rows.append({"mcs": mcs, "ms": ms, "method": method, "n_clusters": len(set(labels) - {-1}),
                         "noise": float((labels == -1).mean())})
    return pd.DataFrame(rows)


def candidate_specs(sweep, k_min, k_max):
    """Strategy A = natural HDBSCAN cluster counts in range; Strategy B = reduce one over-clustered base model."""
    specs = []
    a = (sweep[(sweep.n_clusters >= k_min) & (sweep.n_clusters <= k_max)]
         .sort_values(["n_clusters", "noise"]).drop_duplicates("n_clusters"))
    for r in thin([r for _, r in a.iterrows()], MAX_CANDIDATES_A):
        specs.append({"strategy": "A (HDBSCAN tuning)", "kind": "hdb", "mcs": int(r["mcs"]), "ms": int(r["ms"]),
                      "method": r.get("method", "eom")})

    ok = sweep[(sweep.n_clusters > k_max) & (sweep.noise <= 0.5)]
    base = ok.sort_values("mcs").iloc[0] if not ok.empty else \
        sweep.sort_values(["n_clusters", "noise"], ascending=[False, True]).iloc[0]
    if base["n_clusters"] >= k_min + 1:
        targets = thin(range(k_min, min(k_max + 1, int(base["n_clusters"])) + 1), MAX_CANDIDATES_B)
        for nr in targets:
            specs.append({"strategy": "B (c-TF-IDF reduction)", "kind": "hdb", "mcs": int(base["mcs"]),
                          "ms": int(base["ms"]), "method": base.get("method", "eom"), "nr": int(nr)})
    return specs


def kmeans_specs(k_min, k_max):
    return [{"strategy": "KMeans fallback", "kind": "km", "k": int(k)} for k in thin(range(k_min, k_max + 1), 6)]


def make_vectorizer(extra_stopwords):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
    # min_df=1: BERTopic fits the vectorizer on one document PER TOPIC, so min_df>1 would
    # delete distinctive words whenever only a few topics exist.
    return CountVectorizer(stop_words=list(ENGLISH_STOP_WORDS) + list(extra_stopwords), ngram_range=(1, 2),
                           min_df=1, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")


def make_cluster(spec):
    if spec["kind"] == "km":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=spec["k"], random_state=SEED, n_init=10)
    from hdbscan import HDBSCAN
    return HDBSCAN(min_cluster_size=spec["mcs"], min_samples=spec["ms"], metric="euclidean",
                   cluster_selection_method=spec.get("method", "eom"))


def fit_bertopic(spec, sentences, embeddings, reduced, embedder, extra_stopwords):
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    tm = BERTopic(embedding_model=embedder, umap_model=PrecomputedUMAP(reduced), hdbscan_model=make_cluster(spec),
                  vectorizer_model=make_vectorizer(extra_stopwords),
                  ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
                  calculate_probabilities=False, verbose=False)
    tm.fit_transform(sentences, embeddings)
    nr = spec.get("nr")
    if nr is not None and nr < len(set(tm.topics_)):
        tm.reduce_topics(sentences, nr_topics=nr)
    return tm


def topic_words(tm, top_n=10):
    """Top unigram keywords per topic (outlier topic excluded) for C_v / TD."""
    out = []
    for tid in sorted(set(tm.topics_) - {-1}):
        ws = [w for w, _ in (tm.get_topic(tid) or []) if " " not in w][:top_n]
        if len(ws) >= 2:
            out.append(ws)
    return out


def build_coherence_corpus(sentences):
    """Binary sentence x word matrix: every sentence is one 'virtual document' (window) for C_v."""
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(binary=True, lowercase=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    X = cv.fit_transform(sentences).tocsc()
    return X, cv.vocabulary_


def coherence_cv(tw, corpus):
    """C_v (Roeder et al. 2015) computed directly, following the definition used by gensim's 'c_v':
    boolean sliding windows (a sentence is shorter than the 110-token window, so one window per sentence),
    NPMI between topic words, then the cosine between each word's NPMI context vector and the vector of the
    whole topic word set; averaged over words and over topics."""
    X, vocab = corpus
    n_docs = X.shape[0]
    eps = 1e-12
    scores = []
    for words in tw:
        idx = list(dict.fromkeys(vocab[w] for w in words if w in vocab))
        if len(idx) < 2:
            continue
        B = X[:, idx].toarray().astype(np.float64)
        p = B.mean(axis=0)
        keep = (p > 0) & (p < 1)
        if keep.sum() < 2:
            continue
        B, p = B[:, keep], p[keep]
        joint = (B.T @ B) / n_docs
        with np.errstate(divide="ignore", invalid="ignore"):
            npmi = np.log((joint + eps) / np.outer(p, p)) / -np.log(joint + eps)
        npmi = np.nan_to_num(npmi, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(npmi, 1.0)
        total = npmi.sum(axis=0)  # context vector of the whole word set
        denom = np.linalg.norm(npmi, axis=1) * np.linalg.norm(total)
        cos = (npmi @ total) / np.where(denom > 0, denom, 1.0)
        scores.append(float(cos.mean()))
    return float(np.mean(scores)) if len(scores) >= 2 else np.nan


def topic_diversity(tw):
    words = [w for t in tw for w in t]
    return len(set(words)) / len(words) if words else np.nan


def choose_best(results, k_min, k_max):
    df = results.dropna(subset=["score"])
    df = df[(df["k"] >= k_min) & (df["k"] <= k_max)]
    if df.empty:
        return None
    return df.sort_values(["score", "coherence"], ascending=False).iloc[0]


def enforce_topics(topics, embeddings, k_target, notes):
    """Return topic ids with no outlier (-1) and at most k_target topics, renumbered 0..K-1 by size."""
    topics = np.array(topics)
    emb = np.asarray(embeddings, dtype=float)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def centroids(ids):
        c = np.vstack([emb[topics == t].mean(axis=0) for t in ids])
        return c / np.linalg.norm(c, axis=1, keepdims=True)

    out = topics == -1  # 1) outliers -> nearest topic centroid
    if out.any():
        ids = sorted(set(topics) - {-1})
        topics[out] = np.array(ids)[(emb[out] @ centroids(ids).T).argmax(axis=1)]
        notes.append(f"Reassigned {int(out.sum())} outlier sentences to their nearest topic.")

    ids, merged = sorted(set(topics)), 0  # 2) safety net: merge smallest topics if still too many
    while len(ids) > k_target:
        sizes = {t: int((topics == t).sum()) for t in ids}
        small = min(sizes, key=sizes.get)
        rest = [t for t in ids if t != small]
        topics[topics == small] = rest[int((centroids([small]) @ centroids(rest).T).argmax())]
        ids, merged = rest, merged + 1
    if merged:
        notes.append(f"Merged {merged} small topic(s) to reach {k_target} topics.")

    order = sorted(set(topics), key=lambda t: -int((topics == t).sum()))  # 3) renumber 0..K-1, largest first
    remap = {t: i for i, t in enumerate(order)}
    return [remap[t] for t in topics]


def build_topic_model(sentences, embeddings, reduced, embedder, extra_stopwords, nlp):
    from bertopic.representation import BaseRepresentation

    n = len(sentences)
    k_min, k_max, mcs_list = adaptive_settings(n)
    notes, errors = [], []

    st.write(f"   searching K between {k_min} and {k_max} for {n} sentences...")
    sweep = pd.DataFrame(columns=["mcs", "ms", "method", "n_clusters", "noise"])
    try:
        sweep = sweep_hdbscan(reduced, mcs_list)
        if sweep["n_clusters"].max() < k_min + 1:  # too coarse -> also try 'leaf' selection (finer clusters)
            sweep = pd.concat([sweep, sweep_hdbscan(reduced, mcs_list, "leaf")], ignore_index=True)
    except Exception as e:
        errors.append(f"HDBSCAN sweep failed: {type(e).__name__}: {e}")
    specs = candidate_specs(sweep, k_min, k_max) if len(sweep) else []
    if len(sweep) and not specs:
        errors.append("HDBSCAN found too few clusters on this data for Strategies A/B "
                      f"(max {int(sweep['n_clusters'].max())} clusters).")

    try:
        coh_corpus = build_coherence_corpus(sentences)
    except Exception as e:
        coh_corpus = None
        errors.append(f"Coherence corpus failed: {type(e).__name__}: {e}")

    def evaluate(spec_list):
        out = []
        for spec in spec_list:
            tag = f"{spec['strategy'][0]} mcs={spec.get('mcs')} nr={spec.get('nr')} k={spec.get('k')}"
            try:
                tm_c = fit_bertopic(spec, sentences, embeddings, reduced, embedder, extra_stopwords)
                tw = topic_words(tm_c)
                k = len(set(tm_c.topics_) - {-1})
                cv = np.nan
                if coh_corpus is not None:
                    try:
                        cv = coherence_cv(tw, coh_corpus)
                    except Exception as e:
                        errors.append(f"C_v failed for {tag}: {type(e).__name__}: {e}")
                td = topic_diversity(tw)
                score = cv * td if np.isfinite(cv) and np.isfinite(td) else np.nan
                out.append({"strategy": spec["strategy"], "k": k, "coherence": cv, "diversity": td,
                            "score": score, "spec": spec})
            except Exception as e:
                errors.append(f"{tag}: {type(e).__name__}: {e}")
        return out

    rows = evaluate(specs)
    tuning = pd.DataFrame(rows, columns=["strategy", "k", "coherence", "diversity", "score", "spec"])
    best = choose_best(tuning, k_min, k_max) if len(tuning) else None
    if best is None:  # second pass: KMeans over the K range
        rows += evaluate(kmeans_specs(k_min, k_max))
        tuning = pd.DataFrame(rows, columns=["strategy", "k", "coherence", "diversity", "score", "spec"])
        best = choose_best(tuning, k_min, k_max) if len(tuning) else None
        if best is not None:
            notes.append("HDBSCAN-based candidates could not be scored on this data, so KMeans over the K range was used "
                         "(see 'Choice of topic number' for the reasons).")

    chosen = None
    if best is not None:
        spec, k_sel = best["spec"], int(best["k"])
        chosen = {"strategy": best["strategy"], "k": k_sel, "coherence": float(best["coherence"]),
                  "diversity": float(best["diversity"]), "score": float(best["score"])}
        notes.append(f"Number of topics chosen automatically: K={k_sel} "
                     f"(C_v={best['coherence']:.3f}, TD={best['diversity']:.3f}, strategy {best['strategy'][0]}).")
    else:
        k_mid = int(round((k_min + k_max) / 2))
        spec = {"strategy": "KMeans fallback", "kind": "km", "k": k_mid}
        notes.append(f"C_v x TD could not be computed; used KMeans with K={k_mid} (see 'Choice of topic number').")

    tm = fit_bertopic(spec, sentences, embeddings, reduced, embedder, extra_stopwords)
    topics = enforce_topics(list(tm.topics_), embeddings, len(set(tm.topics_) - {-1}), notes)

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

    from bertopic.vectorizers import ClassTfidfTransformer
    tm.update_topics(sentences, topics=topics, vectorizer_model=make_vectorizer(extra_stopwords),
                     ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
                     representation_model=LemmaDedup(), top_n_words=50)
    return tm, topics, notes, tuning.drop(columns="spec"), (k_min, k_max), chosen, errors


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
    n_asp = df.drop_duplicates(["topic_label", "sentence_id", "aspect_keyword"]).groupby("topic_label").size()
    agg = pd.DataFrame({
        "n_aspects": n_asp,
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
def run_pipeline(texts, extra_stopwords):
    nlp, embedder = load_spacy(), load_embedder()
    blocked = frozenset(extra_stopwords)

    st.write("1/6 Splitting reviews into sentences and finding aspect words...")
    texts = [re.sub(r"\s+", " ", str(t)).strip() for t in texts]
    texts = list(dict.fromkeys(t for t in texts if t))  # drop blanks + exact duplicates
    sent_df = split_sentences(texts, nlp, blocked)
    notes = []
    if len(sent_df) > MAX_SENTENCES:
        sent_df = sent_df.sample(MAX_SENTENCES, random_state=SEED).sort_values("sentence_id").reset_index(drop=True)
        sent_df["sentence_id"] = sent_df.index
        notes.append(f"Corpus capped to a random sample of {MAX_SENTENCES} sentences for demo speed.")
    if len(sent_df) < MIN_SENTENCES:
        raise ValueError(f"Only {len(sent_df)} usable sentences found; need at least {MIN_SENTENCES}.")
    if sent_df["aspects"].map(len).sum() == 0:  # no opinion-bearing nouns anywhere -> fall back to plain nouns
        sent_df["aspects"] = sent_df["aspects_any"]
        notes.append("No opinion-bearing aspect words were detected, so plain nouns were used as aspects.")
    n_scored = int((sent_df["aspects"].map(len) > 0).sum())
    notes.append(f"{n_scored} of {len(sent_df)} sentences contain an opinion-bearing aspect word and were scored for sentiment.")

    sentences = sent_df["sentence"].tolist()
    st.write(f"2/6 Embedding {len(sentences)} sentences...")
    embeddings = embedder.encode(sentences, batch_size=64, show_progress_bar=False, convert_to_numpy=True)

    st.write("3/6 Reducing dimensions (UMAP)...")
    reduced = reduce_embeddings(embeddings)

    st.write("4/6 Choosing the number of topics (C_v x TD) and fitting BERTopic...")
    tm, topics, tnotes, tuning, k_range, chosen, errors = build_topic_model(
        sentences, embeddings, reduced, embedder, extra_stopwords, nlp)
    notes += tnotes
    topic_ids = sorted(set(topics))
    labels = label_topics(tm, topic_ids)
    sent_df["topic_id"] = topics
    sent_df["topic_label"] = sent_df["topic_id"].map(labels)

    st.write("5/6 Scoring aspect-level sentiment...")
    absa = build_absa_frame(sent_df, topics, tm, topic_ids, embeddings, embedder, labels)
    if absa.empty:
        raise ValueError("No aspect terms could be extracted from this text.")
    bar = st.progress(0.0)
    absa = run_absa(absa, progress_cb=bar.progress)

    st.write("6/6 Aggregating per topic...")
    absa, agg = aggregate(absa)
    all_labels = [labels[t] for t in topic_ids]  # keep topics that ended up with no scored sentences
    agg = agg.reindex(all_labels)
    agg.index.name = "topic_label"
    agg["n_aspects"] = agg["n_aspects"].fillna(0).astype(int)
    agg = agg.sort_values("weighted_mean_score", na_position="first")
    empty = [lab for lab in all_labels if agg.loc[lab, "n_aspects"] == 0]
    if empty:
        notes.append(f"{len(empty)} topic(s) had no opinion-bearing aspect words, so no sentiment is shown for them: "
                     + "; ".join(empty))

    keywords = {t: tm.get_topic(t)[:100] for t in topic_ids}
    return {"absa": absa, "agg": agg, "sent_df": sent_df, "labels": labels, "keywords": keywords,
            "tuning": tuning, "k_range": k_range, "chosen": chosen, "tuning_errors": errors,
            "n_topics": len(topic_ids),
            "n_reviews": len(texts), "n_sentences": len(sentences), "notes": notes}


# ---------------------------------------------------------------------------
# Charts (heights scale with the number of topics so few-topic charts stay readable)
# ---------------------------------------------------------------------------
def _bar_height(n):
    return max(420, 150 + 60 * n)


def _y_labels(d):
    """Topic labels; topics without any scored sentence stay visible and say so."""
    return [lab if pd.notna(v) else f"{lab} — no scored sentences"
            for lab, v in zip(d["topic_label"], d["weighted_mean_score"])]


def fig_keywords(res, n_words=10):
    ids = list(res["labels"])
    n = len(ids)
    cols = min(3, n)
    rows = -(-n // cols)
    counts = res["sent_df"]["topic_id"].value_counts()
    titles = [f'{res["labels"][t]}<br><sup>{int(counts.get(t, 0))} sentences</sup>' for t in ids]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, horizontal_spacing=0.16,
                        vertical_spacing=min(0.14, 0.7 / rows))
    for i, t in enumerate(ids):
        words = res["keywords"][t][:n_words][::-1]  # highest score drawn at the top
        fig.add_trace(go.Bar(x=[s for _, s in words], y=[w for w, _ in words], orientation="h",
                             marker_color="#378ADD", showlegend=False,
                             hovertemplate="%{y}: %{x:.3f}<extra></extra>"),
                      row=i // cols + 1, col=i % cols + 1)
    fig.update_layout(title=f"Top {n_words} keywords per topic (c-TF-IDF score)",
                      height=max(420, 340 * rows), margin=dict(l=10, r=10, t=100, b=30))
    fig.update_yaxes(automargin=True)
    fig.update_annotations(font_size=12)
    return fig


def fig_mean_sentiment(agg):
    d = agg.reset_index()
    ylab = _y_labels(d)
    fig = go.Figure(go.Bar(
        x=d["weighted_mean_score"], y=ylab, orientation="h",
        marker=dict(color=d["weighted_mean_score"], colorscale="RdYlGn", cmin=-1, cmax=1,
                    line=dict(color="rgba(0,0,0,0.35)", width=1)),
        text=d["weighted_mean_score"].round(2), textposition="outside", cliponaxis=False,
        customdata=d["n_aspects"], hovertemplate="%{y}<br>mean sentiment: %{x:.3f}<br>scored aspects: %{customdata}<extra></extra>"))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_xaxes(range=[-1.1, 1.1], title="Weighted mean sentiment score")
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=ylab, automargin=True)
    fig.update_layout(title="Mean sentiment by topic (-1 negative … +1 positive)", bargap=0.35,
                      height=_bar_height(len(d)), margin=dict(l=10, r=50, t=70, b=50))
    return fig


def fig_stacked(agg):
    d = agg.reset_index()
    ylab = _y_labels(d)
    fig = go.Figure()
    for col, name, color in [("pct_negative", "negative", "#E24B4A"),
                             ("pct_neutral", "neutral", "#B4B2A9"),
                             ("pct_positive", "positive", "#639922")]:
        vals = d[col].fillna(0)
        fig.add_bar(y=ylab, x=vals, name=name, orientation="h", marker_color=color,
                    text=[f"{v:.1f}%" if v > 4 else "" for v in vals], textposition="inside")
    fig.update_xaxes(range=[0, 100], title="%")
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=ylab, automargin=True)
    fig.update_layout(barmode="stack", title="Sentiment distribution by topic (%)", bargap=0.35,
                      legend=dict(orientation="h", y=-0.2), height=_bar_height(len(d)) + 30,
                      margin=dict(l=10, r=20, t=70, b=70))
    return fig


def fig_tuning(tuning, chosen_k):
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Coherence (C_v)", "Diversity (TD)", "Combined (C_v × TD)"])
    few = tuning["k"].nunique() <= 4  # grouped bars read better than lines when there are only a few K values
    for strat, sub in tuning.groupby("strategy"):
        sub = sub.sort_values("k")
        for j, col in enumerate(["coherence", "diversity", "score"], 1):
            if few:
                tr = go.Bar(x=sub["k"], y=sub[col], name=strat, legendgroup=strat, showlegend=(j == 1),
                            text=sub[col].round(3), textposition="outside")
            else:
                tr = go.Scatter(x=sub["k"], y=sub[col], mode="lines+markers", marker=dict(size=9), name=strat,
                                legendgroup=strat, showlegend=(j == 1))
            fig.add_trace(tr, row=1, col=j)
    for j in (1, 2, 3):
        fig.add_vline(x=chosen_k, line_dash="dash", line_color="gray", row=1, col=j)
    fig.update_xaxes(title_text="number of topics (k)", dtick=1)
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(height=420, barmode="group", legend=dict(orientation="h", y=-0.3),
                      margin=dict(l=10, r=10, t=60, b=40))
    return fig


def wordcloud_image(freqs):
    from wordcloud import WordCloud
    return WordCloud(background_color="white", width=800, height=500, max_words=100,
                     colormap="viridis").generate_from_frequencies(dict(freqs)).to_array()


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------
def parse_stopwords(text):
    return sorted({w for part in re.split(r"[,;\n]+", text.lower()) for w in part.split() if w})


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
               "Topics, their number and sentiment are found automatically.")

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

    sb.header("2. Stop words (optional)")
    extra_sw = parse_stopwords(sb.text_input("Words to ignore (comma-separated)",
                                             placeholder="e.g. brand name, city name"))

    if sb.button("Analyze", type="primary", disabled=not texts):
        try:
            with st.status("Running pipeline...", expanded=True) as status:
                st.session_state["result"] = run_pipeline(texts, extra_sw)
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
    c3.metric("Topics (auto)", res["n_topics"])

    t_over, t_cloud, t_k, t_ex, t_data = st.tabs(
        ["Overview", "Word clouds", "Choice of topic number", "Example sentences", "Data"])

    with t_over:
        st.plotly_chart(fig_keywords(res), use_container_width=True)
        st.plotly_chart(fig_mean_sentiment(res["agg"]), use_container_width=True)
        st.plotly_chart(fig_stacked(res["agg"]), use_container_width=True)

    with t_cloud:
        ids = list(res["labels"])
        for r in range(0, len(ids), 3):
            cols = st.columns(3)
            for c, tid in zip(cols, ids[r:r + 3]):
                c.image(wordcloud_image(res["keywords"][tid]), caption=res["labels"][tid])

    with t_k:
        ch = res.get("chosen")
        if ch:
            st.success(f"Chosen K = {ch['k']} (strategy {ch['strategy']}): C_v = {ch['coherence']:.3f}, "
                       f"TD = {ch['diversity']:.3f}, C_v × TD = {ch['score']:.3f}")
        st.write(f"Search range for this data size: K = {res['k_range'][0]} to {res['k_range'][1]}. "
                 "The K with the highest C_v × TD is used.")
        if len(res["tuning"]):
            st.plotly_chart(fig_tuning(res["tuning"], res["n_topics"]), use_container_width=True)
            tbl = res["tuning"].sort_values("k").round(4)
            st.dataframe(tbl, hide_index=True, use_container_width=True)
        else:
            st.info("No candidate models could be scored; a default K was used.")
        if res.get("tuning_errors"):
            with st.expander(f"Why some candidates were skipped ({len(res['tuning_errors'])})"):
                for e in res["tuning_errors"][:30]:
                    st.code(e, language=None)

    with t_ex:
        topic = st.selectbox("Topic", list(res["agg"].index[::-1]))
        d = res["absa"][res["absa"]["topic_label"] == topic].drop_duplicates(["sentence_id", "aspect_keyword"])
        d = d[["sentence", "aspect_keyword", "sentiment", "sentiment_score"]].sort_values("sentiment_score")
        if d.empty:
            st.info("This topic has no sentences with an opinion-bearing aspect word.")
        else:
            a, b = st.columns(2)
            a.subheader("Most negative")
            a.dataframe(d.head(8), hide_index=True, use_container_width=True)
            b.subheader("Most positive")
            b.dataframe(d.tail(8).iloc[::-1], hide_index=True, use_container_width=True)

    with t_data:
        show = res["agg"].round(3).copy()
        show.insert(0, "sentences", res["sent_df"]["topic_label"].value_counts().reindex(show.index).fillna(0).astype(int))
        st.dataframe(show, use_container_width=True)
        st.download_button("Download topic summary (CSV)", show.to_csv().encode(), "topic_summary.csv", "text/csv")
        st.download_button("Download full results (CSV)", res["absa"].to_csv(index=False).encode(),
                           "absa_results.csv", "text/csv")


if __name__ == "__main__":
    main()
