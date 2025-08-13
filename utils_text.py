import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

DEFAULT_KEYWORDS = [
    r"lawsuit", r"merger", r"acquisition", r"sec\b", r"probe", r"recall",
    r"delivery", r"guidance", r"earnings", r"beats?", r"miss(es)?", r"strike",
    r"autopilot", r"full self[- ]driving", r"battery", r"price (cut|hike)",
]

def _clean(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9%$+\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def collapse_titles_by_day(articles_df: pd.DataFrame, tz="America/New_York"):
    if articles_df is None or len(articles_df) == 0:
        return pd.DataFrame(columns=["Date", "doc"])
    df = articles_df.copy()
    ts = pd.to_datetime(df["publishedAt"], utc=True, errors="coerce").dt.tz_convert(tz)
    df["Date"] = ts.dt.normalize().dt.date
    df["title"] = df["title"].fillna("").map(_clean)
    agg = df.groupby("Date")["title"].apply(lambda x: " . ".join(x)).reset_index(name="doc")
    return agg

def fit_text_pipeline(docs_df: pd.DataFrame, n_topics=20, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=2)
    X = vectorizer.fit_transform(docs_df["doc"])
    n_comp = max(1, min(n_topics, X.shape[1]-1 if X.shape[1] > 1 else 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Z = svd.fit_transform(X)
    topic_cols = [f"topic_{i+1}" for i in range(Z.shape[1])]
    topics = pd.DataFrame(Z, columns=topic_cols, index=docs_df["Date"])
    return vectorizer, svd, topics

def transform_text_pipeline(docs_df: pd.DataFrame, vectorizer, svd):
    if docs_df is None or len(docs_df) == 0:
        return pd.DataFrame(columns=[f"topic_{i+1}" for i in range(svd.n_components)], index=[])
    X = vectorizer.transform(docs_df["doc"])
    Z = svd.transform(X)
    topic_cols = [f"topic_{i+1}" for i in range(Z.shape[1])]
    topics = pd.DataFrame(Z, columns=topic_cols, index=docs_df["Date"])
    return topics

def keyword_counts_by_day(articles_df: pd.DataFrame, patterns=DEFAULT_KEYWORDS, tz="America/New_York"):
    if articles_df is None or len(articles_df) == 0:
        return pd.DataFrame(columns=["Date"])
    df = articles_df.copy()
    ts = pd.to_datetime(df["publishedAt"], utc=True, errors="coerce").dt.tz_convert(tz)
    df["Date"] = ts.dt.normalize().dt.date
    df["title"] = df["title"].fillna("").map(_clean)

    out = df.groupby("Date")["title"].apply(list).reset_index()
    for pat in patterns:
        col = f"kw_{re.sub('[^a-z]+','_', pat)}"
        out[col] = out["title"].apply(lambda L: sum(bool(re.search(pat, t)) for t in L))
    return out.drop(columns=["title"]).set_index("Date")

def build_daily_text_features(articles_df, vectorizer=None, svd=None, n_topics=20):
    docs = collapse_titles_by_day(articles_df)
    docs = docs.set_index("Date")
    if vectorizer is None or svd is None:
        vectorizer, svd, topics = fit_text_pipeline(docs.reset_index(), n_topics=n_topics)
        fitted = True
    else:
        topics = transform_text_pipeline(docs.reset_index(), vectorizer, svd)
        fitted = False
    kw = keyword_counts_by_day(articles_df).reindex(topics.index).fillna(0)
    feats = topics.join(kw, how="outer").fillna(0)
    return feats, vectorizer, svd, fitted
