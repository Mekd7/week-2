# src/themes.py
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
import joblib

# Optional: spaCy for lemmatization if you want
try:
    import spacy
    SPACY_AVAILABLE = True
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except Exception:
    SPACY_AVAILABLE = False
    nlp = None

class ThemeExtractor:
    def __init__(self,
                 ngram_range=(1,2),
                 top_k_keywords=30,
                 min_df=3):
        """
        ngram_range: TF-IDF ngram range
        top_k_keywords: how many candidate keywords to extract per bank
        min_df: minimal document frequency for TF-IDF
        """
        self.ngram_range = ngram_range
        self.top_k = top_k_keywords
        self.min_df = min_df
        self.vectorizer = None

    @staticmethod
    def _clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def fit_tfidf(self, texts: List[str]):
        cleaned = [self._clean_text(t) for t in texts]
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, min_df=self.min_df, stop_words='english')
        self.X = self.vectorizer.fit_transform(cleaned)
        return self.vectorizer

    def top_keywords_for_bank(self, texts: List[str], top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        if top_n is None:
            top_n = self.top_k
        if self.vectorizer is None:
            self.fit_tfidf(texts)
        cleaned = [self._clean_text(t) for t in texts]
        X = self.vectorizer.transform(cleaned)
        # tf-idf summed over docs
        scores = np.asarray(X.sum(axis=0)).ravel()
        feat_names = np.array(self.vectorizer.get_feature_names_out())
        top_idx = np.argsort(scores)[::-1][:top_n]
        return list(zip(feat_names[top_idx], scores[top_idx]))

    def extract_bank_keywords(self, df: pd.DataFrame, bank_col: str = "bank", text_col: str = "review") -> Dict[str, List[Tuple[str,float]]]:
        result = {}
        for bank, group in df.groupby(bank_col):
            texts = group[text_col].astype(str).tolist()
            # fit vectorizer fresh for each bank to prioritize bank-specific keywords
            self.vectorizer = None
            self.fit_tfidf(texts)
            result[bank] = self.top_keywords_for_bank(texts)
        return result

    def rule_based_theme_mapping(self, keywords: List[str], mapping: Dict[str, List[str]]) -> List[str]:
        """
        mapping: dict theme_name -> list of keyword substrings
        returns list of theme names that matched
        """
        matched = set()
        for kw in keywords:
            k = kw.lower()
            for theme, substrs in mapping.items():
                for s in substrs:
                    if s.lower() in k:
                        matched.add(theme)
        return list(matched)

    def cluster_keywords(self, keywords: List[str], n_clusters: int = 3) -> Dict[int, List[str]]:
        """
        fallback clustering: cluster keyword vectors (TF-IDF of keywords) into n_clusters
        returns dict cluster_idx -> list of keywords
        """
        if not keywords:
            return {}
        vect = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
        X = vect.fit_transform(keywords)
        # optionally reduce dimensionality if too large
        if X.shape[1] > 50:
            svd = TruncatedSVD(n_components=min(20, X.shape[1]-1))
            model = make_pipeline(svd, KMeans(n_clusters=n_clusters, random_state=42))
        else:
            model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(X)
        labels = model[-1].labels_ if hasattr(model, '__len__') else model.labels_
        clusters = {}
        for kw, lab in zip(keywords, labels):
            clusters.setdefault(int(lab), []).append(kw)
        return clusters
