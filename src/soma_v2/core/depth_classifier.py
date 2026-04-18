"""
SOMA V2 — Depth Classifier
===========================
Predicts task depth (simple / medium / complex) from task text + routing metadata.

Feature set (two groups):
  Text features  — extracted from the raw event string:
    text_len         : character count (longer = more complex)
    word_count       : token count
    complex_kw       : count of complexity-signalling keywords
    simple_kw        : count of simplicity-signalling keywords
    multi_entity     : 1 if text mentions multiple entities/nodes/drones
    has_coordinate   : 1 if task involves multi-agent coordination
    has_optimise     : 1 if task is an optimisation goal
    has_status_check : 1 if task is a simple status/info query

  Metadata features — routing context:
    agent_role       : EMERGENCY=3, SUPERVISOR=2, PEER=1, ROUTINE=0
    confidence       : float 0–1
    urgency          : low=0, medium=1, high=2
    contested        : 0/1
    reroute_attempts : int

Text features dominate (they contain far more signal than metadata).
Metadata provides disambiguation when text is ambiguous.

Online retraining: every `retrain_every` labelled outcomes the classifier
retrains in a background thread (same V9 approach, extended to text features).
"""

import logging
import os
import re
import threading
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ROLE_ORD    = {"EMERGENCY": 3, "SUPERVISOR": 2, "PEER": 1, "ROUTINE": 0}
URGENCY_ORD = {"low": 0, "medium": 1, "high": 2}

DEPTH_SIMPLE  = "simple"
DEPTH_MEDIUM  = "medium"
DEPTH_COMPLEX = "complex"

# ── keyword lists ─────────────────────────────────────────────────────────────

_COMPLEX_KW = re.compile(
    r"\b(coordinat|mission|rescue|multi|recalibrat|optimis|optimiz|"
    r"multi.node|southern grid|network|energy distribution|"
    r"multi.drone|across|full.scale|strategic|emergency response)\b",
    re.IGNORECASE,
)
_SIMPLE_KW = re.compile(
    r"\b(check|status|ping|report|query|list|show|display|"
    r"get|fetch|read|monitor|look up|what is)\b",
    re.IGNORECASE,
)
_MULTI_ENTITY = re.compile(
    r"\b(\d[-\s]?drone|\d[-\s]?node|multiple|several|all|swarm|fleet|grid)\b",
    re.IGNORECASE,
)
_COORDINATE = re.compile(
    r"\b(coordinat|orchestrat|synchr|allocat|assign.*team|dispatch.*team)\b",
    re.IGNORECASE,
)
_OPTIMISE = re.compile(
    r"\b(optimis|optimiz|balanc|distribut|schedul|allocat.*resource)\b",
    re.IGNORECASE,
)
_STATUS_CHECK = re.compile(
    r"\b(status|health|ping|alive|check.*of|state of)\b",
    re.IGNORECASE,
)

TEXT_FEATURES = [
    "text_len", "word_count",
    "complex_kw", "simple_kw",
    "multi_entity", "has_coordinate", "has_optimise", "has_status_check",
]
META_FEATURES = ["agent_role", "confidence", "urgency", "contested", "reroute_attempts"]
ALL_FEATURES  = TEXT_FEATURES + META_FEATURES


def _text_features(text: str) -> dict:
    t = text or ""
    return {
        "text_len":        len(t),
        "word_count":      len(t.split()),
        "complex_kw":      len(_COMPLEX_KW.findall(t)),
        "simple_kw":       len(_SIMPLE_KW.findall(t)),
        "multi_entity":    int(bool(_MULTI_ENTITY.search(t))),
        "has_coordinate":  int(bool(_COORDINATE.search(t))),
        "has_optimise":    int(bool(_OPTIMISE.search(t))),
        "has_status_check":int(bool(_STATUS_CHECK.search(t))),
    }


def _rule_classify(text: str) -> Optional[str]:
    """
    Fast keyword rule — fires before the ML model for high-confidence cases.
    Returns None if ambiguous (let the model decide).
    """
    tf = _text_features(text)
    # Unambiguously simple: only simple keywords, no complex signals, short text
    if (tf["simple_kw"] >= 1 and tf["complex_kw"] == 0
            and tf["multi_entity"] == 0 and tf["has_coordinate"] == 0
            and tf["has_optimise"] == 0 and tf["word_count"] <= 12):
        return DEPTH_SIMPLE
    # Unambiguously complex: has coordination or optimisation AND multi-entity signal
    if (tf["has_coordinate"] or tf["has_optimise"] or tf["multi_entity"]) and tf["complex_kw"] >= 1:
        return DEPTH_COMPLEX
    return None  # ambiguous — use model


class DepthClassifier:
    """
    Text-aware depth classifier.

    predict() now accepts the event text string alongside metadata.
    A fast keyword rule fires first; the ML model handles ambiguous cases.
    Falls back to DEPTH_MEDIUM if the model is unavailable.

    Online retraining accumulates (text, metadata, depth) tuples and
    retrains a RandomForest on ALL_FEATURES every `retrain_every` outcomes.
    """

    def __init__(
        self,
        model_path: str,
        base_csv: str,
        min_confidence: float = 0.60,
        retrain_every: int = 50,
    ):
        self.model_path     = model_path
        self.base_csv       = base_csv
        self.min_confidence = min_confidence
        self.retrain_every  = retrain_every

        self._clf:        object = None
        self._available:  bool   = False
        self._lock               = threading.Lock()
        self._buffer:     list   = []
        self._generation: int    = 0
        self._retraining: bool   = False

        self._rule_hits:  int    = 0
        self._model_hits: int    = 0
        self._fallback_hits: int = 0

        self._load()

    def _load(self) -> None:
        try:
            import joblib
            if os.path.exists(self.model_path):
                self._clf       = joblib.load(self.model_path)
                self._available = True
                logger.info(f"DepthClassifier: model loaded from {self.model_path}")
            else:
                logger.info("DepthClassifier: no saved model — using rule-based until first retrain")
        except ImportError:
            logger.warning("DepthClassifier: joblib not installed")

    # ── inference ────────────────────────────────────────────────────────────

    def predict(
        self,
        agent_role: str,
        confidence: float,
        urgency,
        contested: bool,
        reroute_attempts: int,
        event_text: str = "",
    ) -> Tuple[str, float]:
        """
        Returns (depth, probability).

        Priority:
          1. Keyword rule — O(1), high confidence cases
          2. ML model    — handles ambiguous cases with text+meta features
          3. Fallback    — DEPTH_MEDIUM when model unavailable
        """
        # 1. Keyword rule
        rule_result = _rule_classify(event_text)
        if rule_result is not None:
            self._rule_hits += 1
            logger.debug(f"DepthClassifier: rule={rule_result} text='{event_text[:50]}'")
            return rule_result, 1.0

        # 2. ML model
        if self._available and self._clf is not None:
            try:
                tf  = _text_features(event_text)
                row = np.array([[
                    tf["text_len"],
                    tf["word_count"],
                    tf["complex_kw"],
                    tf["simple_kw"],
                    tf["multi_entity"],
                    tf["has_coordinate"],
                    tf["has_optimise"],
                    tf["has_status_check"],
                    ROLE_ORD.get(str(agent_role).upper(), 1),
                    float(confidence),
                    URGENCY_ORD.get(urgency, 1) if isinstance(urgency, str) else int(urgency),
                    1 if contested else 0,
                    int(reroute_attempts),
                ]])
                with self._lock:
                    clf = self._clf
                probs     = clf.predict_proba(row)[0]
                classes   = clf.classes_
                best_idx  = int(probs.argmax())
                best_prob = float(probs[best_idx])
                best_depth = classes[best_idx]

                self._model_hits += 1
                logger.debug(f"DepthClassifier: model={best_depth} p={best_prob:.3f}")
                if best_prob >= self.min_confidence:
                    return best_depth, best_prob
            except Exception as exc:
                logger.error(f"DepthClassifier.predict error: {exc}")

        # 3. Fallback — urgency heuristic
        self._fallback_hits += 1
        urgency_str = urgency if isinstance(urgency, str) else "medium"
        if urgency_str == "high" and reroute_attempts > 0:
            return DEPTH_COMPLEX, 0.0
        return DEPTH_MEDIUM, 0.0

    # ── online learning ──────────────────────────────────────────────────────

    def record_outcome(
        self,
        agent_role: str,
        confidence: float,
        urgency,
        contested: bool,
        reroute_attempts: int,
        true_depth: str,
        event_text: str = "",
    ) -> None:
        tf = _text_features(event_text)
        self._buffer.append({
            **tf,
            "agent_role":       ROLE_ORD.get(str(agent_role).upper(), 1),
            "confidence":       float(confidence),
            "urgency":          URGENCY_ORD.get(urgency, 1) if isinstance(urgency, str) else int(urgency),
            "contested":        1 if contested else 0,
            "reroute_attempts": int(reroute_attempts),
            "depth":            true_depth,
        })
        if len(self._buffer) >= self.retrain_every and not self._retraining:
            self._retraining = True
            threading.Thread(target=self._retrain, daemon=True).start()

    def _retrain(self) -> None:
        try:
            import pandas as pd
            from sklearn.ensemble import RandomForestClassifier

            with self._lock:
                batch = list(self._buffer)
                self._buffer.clear()

            frames = []

            # Seed from V1 CSV if available — derive depth + text features
            if self.base_csv and os.path.exists(self.base_csv):
                base = pd.read_csv(self.base_csv)
                def _depth(row):
                    src = str(row.get("source", ""))
                    lbl = str(row.get("label", ""))
                    if "D1" in src:
                        return DEPTH_SIMPLE
                    elif lbl in ("EXHAUST", "LLM_REROUTE"):
                        return DEPTH_COMPLEX
                    return DEPTH_MEDIUM
                base["depth"] = base.apply(_depth, axis=1)
                # Add dummy text features (V1 CSV has no text column)
                for col in TEXT_FEATURES:
                    if col not in base.columns:
                        base[col] = 0
                if base["agent_role"].dtype == object:
                    base["agent_role"] = base["agent_role"].map(
                        lambda r: ROLE_ORD.get(str(r).upper(), 1)
                    )
                frames.append(base[ALL_FEATURES + ["depth"]].dropna())

            frames.append(pd.DataFrame(batch))
            df = pd.concat(frames, ignore_index=True).dropna(subset=["depth"])

            X = df[ALL_FEATURES].values.astype(float)
            y = df["depth"].values

            clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            clf.fit(X, y)

            import joblib
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(clf, self.model_path)

            with self._lock:
                self._clf         = clf
                self._available   = True
                self._generation += 1

            logger.info(
                f"DepthClassifier retrain complete: gen={self._generation} "
                f"samples={len(df)} features={ALL_FEATURES}"
            )

        except Exception as exc:
            logger.error(f"DepthClassifier retrain failed: {exc}", exc_info=True)
        finally:
            self._retraining = False

    # ── introspection ────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._available

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def stats(self) -> dict:
        total = self._rule_hits + self._model_hits + self._fallback_hits or 1
        return {
            "generation":     self._generation,
            "buffer_pending": len(self._buffer),
            "retrain_every":  self.retrain_every,
            "available":      self._available,
            "rule_hits":      self._rule_hits,
            "model_hits":     self._model_hits,
            "fallback_hits":  self._fallback_hits,
            "rule_hit_rate":  round(self._rule_hits / total, 3),
        }
