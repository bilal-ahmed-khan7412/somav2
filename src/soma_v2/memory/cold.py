"""
SOMA V2 — Cold Memory Layer (ChromaDB)
========================================
Episodic memory: stores past task outcomes as embeddings for semantic recall.
Only queried when an agent explicitly needs "have we seen something like this?"

ChromaDB is optional — if not installed, ColdMemory degrades gracefully to
a flat in-process list with linear text search (good enough for dev/testing).

Schema per document
-------------------
  id       : unique episode id  (uuid hex)
  document : event text (embedded by Chroma's default embedding fn)
  metadata : {agent_id, agent_type, action, urgency, success, timestamp}
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger("SOMA_V2.COLD_MEMORY")

# Shared normaliser — same logic as deliberative.py so keys align
_STRIP_PREFIX  = re.compile(r"^var_\d+\s*:\s*", re.IGNORECASE)
_STRIP_NUMBERS = re.compile(r"\b[A-Z]?\d+\b")
_NORM_AGENTS   = re.compile(r"\b(drone|rov|submersible|uav|sub)\b", re.IGNORECASE)
_NORM_ACTIONS  = re.compile(r"\b(rescue|extraction|salvage|mission)\b", re.IGNORECASE)
_NORM_SENSORS  = re.compile(r"\b(sonar|sensor|radar)\b", re.IGNORECASE)
_MULTI_SPACE   = re.compile(r"\s{2,}")

def _norm(text: str) -> str:
    text = _STRIP_PREFIX.sub("", text)
    text = _STRIP_NUMBERS.sub("N", text)
    text = _NORM_AGENTS.sub("AGENT_UNIT", text)
    text = _NORM_ACTIONS.sub("MISSION", text)
    text = _NORM_SENSORS.sub("SENSOR", text)
    return _MULTI_SPACE.sub(" ", text).lower().strip()

_CHROMA_AVAILABLE = False
try:
    import chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    pass

import importlib.util as _ilu
_ST_AVAILABLE = _ilu.find_spec("sentence_transformers") is not None
del _ilu

# Model is loaded lazily inside _STEmbeddingFunction.__call__ the first time
# ChromaDB actually needs an embedding — never at import time.
_st_model = None


def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("ColdMemory: loading sentence-transformers all-MiniLM-L6-v2 ...")
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("ColdMemory: embedding model ready")
    return _st_model


try:
    from chromadb import EmbeddingFunction
    _EF_BASE = EmbeddingFunction
except Exception:
    _EF_BASE = object


class _STEmbeddingFunction(_EF_BASE):
    """ChromaDB-compatible embedding function backed by sentence-transformers."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        model = _get_st_model()
        vecs = model.encode(input, normalize_embeddings=True)
        return vecs.tolist()


# ── fallback store (no ChromaDB) ─────────────────────────────────────────────

class _FlatStore:
    """
    Linear-scan fallback when chromadb is not installed.

    Similarity is computed on *normalised* text (Var_N: prefixes and bare
    integers stripped) so that "Var_3: Coordinate rescue in Sector 7" and
    "Var_9: Coordinate rescue in Sector 12" score as near-identical.

    Score = Jaccard coefficient on normalised token sets, so short queries
    don't get unfairly penalised against long stored texts.
    """

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        self._episodes: List[Dict] = []
        self._file_path = None
        if persist_dir:
            import os
            os.makedirs(persist_dir, exist_ok=True)
            self._file_path = os.path.join(persist_dir, "flat_memory.jsonl")
            self._load()

    def _load(self) -> None:
        import os, json
        if not os.path.exists(self._file_path):
            return
        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self._episodes.append(json.loads(line))
            logger.info(f"ColdMemory: loaded {len(self._episodes)} episodes from {self._file_path}")
        except Exception as exc:
            logger.warning(f"ColdMemory: failed to load flat memory: {exc}")

    def add(self, episode_id: str, text: str, metadata: Dict) -> None:
        norm = _norm(text)
        entry = {"id": episode_id, "text": text, "norm": norm, "meta": metadata}
        self._episodes.append(entry)
        if self._file_path:
            import json
            try:
                with open(self._file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as exc:
                logger.warning(f"ColdMemory: failed to save flat memory: {exc}")

    def query(self, query_text: str, n: int) -> List[Dict]:
        q_tokens = set(_norm(query_text).split())
        if not q_tokens:
            return []
        scored = []
        for ep in self._episodes:
            ep_tokens = set(ep.get("norm", ep["text"].lower()).split())
            inter     = len(q_tokens & ep_tokens)
            if inter == 0:
                continue
            union = len(q_tokens | ep_tokens)
            jaccard = inter / union if union else 0.0
            scored.append((jaccard, ep))
        scored.sort(key=lambda x: -x[0])
        return [ep for _, ep in scored[:n]]

    @property
    def count(self) -> int:
        return len(self._episodes)


# ── cold memory ───────────────────────────────────────────────────────────────

class ColdMemory:
    """
    Episodic memory backed by ChromaDB (or flat fallback).

    Parameters
    ----------
    persist_dir : directory for ChromaDB on-disk persistence
                  (None = in-memory Chroma client)
    collection  : ChromaDB collection name
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection: str = "soma_episodes",
    ) -> None:
        self._collection_name = collection
        self._chroma          = None
        self._collection      = None
        self._fallback        = None
        self._write_count     = 0

        if _CHROMA_AVAILABLE:
            try:
                ef = _STEmbeddingFunction() if _ST_AVAILABLE else None
                if persist_dir:
                    self._chroma = chromadb.PersistentClient(path=persist_dir)
                else:
                    self._chroma = chromadb.Client()
                if ef:
                    self._collection = self._chroma.get_or_create_collection(
                        collection, embedding_function=ef
                    )
                else:
                    self._collection = self._chroma.get_or_create_collection(collection)
                # Health probe — verifies embedding function works end-to-end.
                _probe_id = "__probe__"
                self._collection.upsert(ids=[_probe_id], documents=["probe"],
                                        metadatas=[{"probe": 1}])
                self._collection.query(query_texts=["probe"], n_results=1)
                self._collection.delete(ids=[_probe_id])
                backend_tag = "sentence-transformers" if ef else "ONNX-default"
                logger.info(f"ColdMemory: ChromaDB ready (collection='{collection}', embed={backend_tag})")
            except Exception as exc:
                logger.warning(f"ColdMemory: ChromaDB unavailable ({exc}), using flat fallback")
                self._chroma     = None
                self._collection = None
                self._fallback   = _FlatStore(persist_dir=persist_dir)
        else:
            logger.info("ColdMemory: chromadb not installed — using flat fallback")
            self._fallback = _FlatStore(persist_dir=persist_dir)

    # ── write ─────────────────────────────────────────────────────────────────

    def record(
        self,
        event: str,
        agent_id: str,
        agent_type: str,
        action: str,
        urgency: str,
        success: bool,
        extra: Optional[Dict] = None,
    ) -> str:
        episode_id = uuid.uuid4().hex
        norm_event = _norm(event)
        metadata   = {
            "agent_id":   agent_id,
            "agent_type": agent_type,
            "action":     action,
            "urgency":    urgency,
            "success":    int(success),
            "timestamp":  time.time(),
            **(extra or {}),
        }

        if self._collection is not None:
            try:
                self._collection.add(
                    ids=[episode_id],
                    documents=[norm_event],
                    metadatas=[metadata],
                )
            except Exception as exc:
                logger.warning(f"ColdMemory.record chroma error: {exc}")
        elif self._fallback is not None:
            self._fallback.add(episode_id, event, metadata)

        self._write_count += 1
        return episode_id

    # ── recall ────────────────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        n: int = 5,
        filter_success: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the n most semantically similar past episodes.
        Returns list of {event, metadata, distance} dicts.
        """
        norm_query = _norm(query)
        if self._collection is not None:
            return self._recall_chroma(norm_query, n, filter_success)
        if self._fallback is not None:
            results = self._fallback.query(norm_query, n)
            if filter_success is not None:
                results = [r for r in results
                           if bool(r["meta"].get("success")) == filter_success]
            # Convert Jaccard score to a distance-like value (1 - jaccard)
            q_tok = set(norm_query.split())
            out   = []
            for r in results:
                ep_tok  = set(r.get("norm", r["text"].lower()).split())
                inter   = len(q_tok & ep_tok)
                union   = len(q_tok | ep_tok)
                jaccard = inter / union if union else 0.0
                out.append({"event": r["text"], "metadata": r["meta"],
                            "distance": round(1.0 - jaccard, 4)})
            return out
        return []

    def _recall_chroma(self, query: str, n: int, filter_success: Optional[bool]) -> List[Dict]:
        where = None
        if filter_success is not None:
            where = {"success": {"$eq": int(filter_success)}}
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n, max(self._collection.count(), 1)),
                where=where,
            )
            episodes = []
            docs  = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                episodes.append({"event": doc, "metadata": meta, "distance": round(dist, 4)})
            return episodes
        except Exception as exc:
            logger.warning(f"ColdMemory.recall chroma error: {exc}")
            return []

    # ── stats ─────────────────────────────────────────────────────────────────

    @property
    def episode_count(self) -> int:
        if self._collection is not None:
            try:
                return self._collection.count()
            except Exception:
                return self._write_count
        if self._fallback is not None:
            return self._fallback.count
        return 0

    @property
    def backend(self) -> str:
        return "chromadb" if self._collection is not None else "flat"

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "backend":       self.backend,
            "episode_count": self.episode_count,
            "writes":        self._write_count,
        }
