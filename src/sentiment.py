# src/sentiment.py
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

# Transformer imports are optional; we gracefully fallback to VADER
try:
    from transformers import pipeline, Pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

class SentimentAnalyzer:
    def __init__(self, method: str = "auto", device: Optional[int] = None):
        """
        method: 'auto' (try transformers -> fallback vader), 'transformer', or 'vader'
        device: int for transformers (e.g., 0 for GPU), None for CPU
        """
        self.method = method
        self.device = device
        self._pipeline: Optional[Pipeline] = None
        self._vader = None

        if self.method in ("auto", "transformer") and TRANSFORMERS_AVAILABLE:
            try:
                # DistilBERT fine-tuned on SST-2
                self._pipeline = pipeline("sentiment-analysis",
                                         model="distilbert-base-uncased-finetuned-sst-2-english",
                                         device=self.device if self.device is not None else -1)
                self.method = "transformer"
            except Exception:
                self._pipeline = None

        if (self.method == "auto" and self._pipeline is None) or self.method == "vader":
            if VADER_AVAILABLE:
                self._vader = SentimentIntensityAnalyzer()
                self.method = "vader"
            else:
                raise RuntimeError("No sentiment backend available. Install transformers or vaderSentiment.")

    def _transformer_predict(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        returns list of (label, score) where label is POSITIVE/NEGATIVE
        we map to positive/negative/neutral using confidence threshold
        """
        assert self._pipeline is not None
        results = []
        # pipeline accepts list
        out = self._pipeline(texts, truncation=True)
        for r in out:
            label = r.get("label", "")
            score = float(r.get("score", 0.0))
            # Map to pos/neg/neutral: if confidence low -> neutral
            if score < 0.60:
                results.append(("neutral", score if label == "POSITIVE" else -score))
            else:
                if label.upper().startswith("POS"):
                    results.append(("positive", score))
                else:
                    results.append(("negative", -score))
        return results

    def _vader_predict(self, texts: List[str]) -> List[Tuple[str, float]]:
        assert self._vader is not None
        results = []
        for t in texts:
            v = self._vader.polarity_scores(t)
            comp = v["compound"]
            # thresholds: standard VADER
            if comp >= 0.05:
                results.append(("positive", float(comp)))
            elif comp <= -0.05:
                results.append(("negative", float(comp)))
            else:
                results.append(("neutral", float(comp)))
        return results

    def analyze_series(self, reviews: pd.Series, batch_size: int = 64) -> pd.DataFrame:
        """
        reviews: pd.Series of text
        returns DataFrame with columns: sentiment_label, sentiment_score
        sentiment_score is positive float for positive, negative float (<=0) for negative, near-0 for neutral
        """
        texts = reviews.fillna("").astype(str).tolist()
        n = len(texts)
        labels = []
        scores = []

        for i in range(0, n, batch_size):
            batch = texts[i:i+batch_size]
            if self.method == "transformer" and self._pipeline is not None:
                preds = self._transformer_predict(batch)
            else:
                preds = self._vader_predict(batch)
            for lab, sc in preds:
                labels.append(lab)
                scores.append(sc)

        return pd.DataFrame({
            "sentiment_label": labels,
            "sentiment_score": scores
        }, index=reviews.index)
