"""
RAG (Retrieval-Augmented Generation) Pipeline for Location Guide App.

Handles embedding, storage, retrieval, and AI-powered recommendation generation
using scikit-learn TF-IDF vectorization. Supports multi-category data including
restaurants, beaches, temples, and malls.
"""

import re
import json
import logging
from typing import Any
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class PlaceDocument:
    """Represents an embedded place document in the RAG pipeline."""

    place_id: str
    name: str
    category: str
    source: str
    raw_data: dict
    embedding_index: int
    text_representation: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


@dataclass
class RAGContext:
    """Container for retrieved context from the RAG pipeline."""

    query: str
    documents: list[PlaceDocument]
    similarity_scores: np.ndarray
    category_filter: str | None = None
    top_k: int = 5


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for location recommendations.

    Uses TF-IDF vectorization for embeddings and cosine similarity for retrieval.
    No external vector database required.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            strip_accents="unicode",
            stop_words="english",
        )
        self.documents: list[PlaceDocument] = []
        self.tfidf_matrix = None
        self.is_fitted = False
        self._category_indices: dict[str, list[int]] = {}

    def process_scraped_data(self, scraped_data: list[dict]) -> list[PlaceDocument]:
        """
        Process raw scraped data from multiple sources into structured documents.

        Args:
            scraped_data: List of dicts with keys:
                - place_id: Unique identifier
                - name: Place name
                - category: One of restaurants/beaches/temples/malls
                - source: Data source identifier
                - data: Raw place data (dict with details, reviews, etc.)

        Returns:
            List of processed PlaceDocument objects.
        """
        processed = []

        for item in scraped_data:
            place_id = item.get("place_id", f"unknown_{len(processed)}")
            name = item.get("name", "Unknown Place")
            category = item.get("category", "general").lower()
            source = item.get("source", "unknown")
            raw_data = item.get("data", {})

            text_repr = self._build_text_representation(name, category, raw_data)

            doc = PlaceDocument(
                place_id=place_id,
                name=name,
                category=category,
                source=source,
                raw_data=raw_data,
                embedding_index=len(self.documents),
                text_representation=text_repr,
                metadata={
                    "processed_at": datetime.utcnow().isoformat(),
                    "source": source,
                    "text_length": len(text_repr),
                },
            )

            self.documents.append(doc)
            processed.append(doc)

            if category not in self._category_indices:
                self._category_indices[category] = []
            self._category_indices[category].append(doc.embedding_index)

        logger.info(
            "Processed %d scraped documents across %d categories",
            len(processed),
            len(self._category_indices),
        )

        return processed

    def embed_text(self, texts: list[str] | None = None) -> np.ndarray:
        """
        Embed text using TF-IDF vectorization.

        If no texts provided, fits the vectorizer on all stored documents.
        If texts are provided, transforms them using the fitted vectorizer.

        Args:
            texts: Optional list of text strings to embed.

        Returns:
            TF-IDF embedding matrix.
        """
        if texts is None:
            corpus = [doc.text_representation for doc in self.documents]
            if not corpus:
                raise ValueError("No documents to embed. Call process_scraped_data first.")

            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            self.is_fitted = True

            for i, doc in enumerate(self.documents):
                doc.embedding_index = i

            logger.info(
                "Fitted TF-IDF on %d documents with %d features",
                len(corpus),
                self.tfidf_matrix.shape[1],
            )

            return self.tfidf_matrix
        else:
            if not self.is_fitted:
                raise RuntimeError(
                    "Vectorizer not fitted. Call embed_text() with no args first."
                )

            query_matrix = self.vectorizer.transform(texts)
            logger.info("Embedded %d query texts", len(texts))
            return query_matrix

    def retrieve_context(
        self,
        query: str,
        category: str | None = None,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> RAGContext:
        """
        Retrieve relevant context based on user query using cosine similarity.

        Args:
            query: User search query.
            category: Optional category filter (restaurants/beaches/temples/malls).
            top_k: Number of top results to return.
            min_similarity: Minimum similarity threshold.

        Returns:
            RAGContext with retrieved documents and scores.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call embed_text() first.")

        query_embedding = self.embed_text([query])

        if category and category in self._category_indices:
            indices = self._category_indices[category]
            category_matrix = self.tfidf_matrix[indices]
            similarities = cosine_similarity(query_embedding, category_matrix).flatten()

            top_local_indices = np.argsort(similarities)[::-1][:top_k]
            top_global_indices = [indices[i] for i in top_local_indices]
            top_scores = similarities[top_local_indices]
        else:
            similarities = cosine_similarity(query_embedding, self.tfidf_matrix).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            top_global_indices = top_indices.tolist()
            top_scores = similarities[top_indices]

        retrieved_docs = []
        filtered_scores = []

        for idx, score in zip(top_global_indices, top_scores):
            if score >= min_similarity:
                doc = self.documents[idx]
                doc.score = float(score)
                retrieved_docs.append(doc)
                filtered_scores.append(score)

        context = RAGContext(
            query=query,
            documents=retrieved_docs,
            similarity_scores=np.array(filtered_scores),
            category_filter=category,
            top_k=top_k,
        )

        logger.info(
            "Retrieved %d documents for query '%s' (category=%s)",
            len(retrieved_docs),
            query,
            category,
        )

        return context

    def generate_recommendation(
        self,
        query: str,
        user_prefs: dict | None = None,
        category: str | None = None,
        time_of_day: str | None = None,
        current_hour: int | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Generate structured AI recommendations based on query and preferences.

        Combines RAG retrieval with AI scoring to produce ranked recommendations.

        Args:
            query: User search query.
            user_prefs: User preference dict (budget, vibe, dietary, etc.).
            category: Optional category filter.
            time_of_day: Time context (morning/afternoon/evening/night).
            current_hour: Current hour (0-23).
            top_k: Number of recommendations.

        Returns:
            Dict with recommendations, metadata, and query analysis.
        """
        from backend.ai.scoring import calculate_ai_score

        context = self.retrieve_context(query, category=category, top_k=top_k * 2)

        if not context.documents:
            return {
                "query": query,
                "recommendations": [],
                "total_results": 0,
                "query_analysis": self._analyze_query(query),
                "metadata": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "pipeline": "rag-tfidf",
                },
            }

        scored_places = []

        for doc in context.documents:
            place_data = {
                "place_id": doc.place_id,
                "name": doc.name,
                "category": doc.category,
                **doc.raw_data,
            }

            ai_score = calculate_ai_score(
                place=place_data,
                user_prefs=user_prefs or {},
                time_of_day=time_of_day,
                current_hour=current_hour or datetime.utcnow().hour,
            )

            combined_score = (ai_score["total_score"] * 0.7) + (doc.score * 30.0)

            scored_places.append(
                {
                    "place_id": doc.place_id,
                    "name": doc.name,
                    "category": doc.category,
                    "rag_similarity": round(float(doc.score), 4),
                    "ai_score": ai_score,
                    "combined_score": round(combined_score, 2),
                    "details": doc.raw_data,
                    "source": doc.source,
                    "highlights": self._extract_highlights(doc, query),
                }
            )

        scored_places.sort(key=lambda x: x["combined_score"], reverse=True)
        top_recommendations = scored_places[:top_k]

        return {
            "query": query,
            "recommendations": top_recommendations,
            "total_results": len(scored_places),
            "query_analysis": self._analyze_query(query),
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "pipeline": "rag-tfidf",
                "category_filter": category,
                "time_context": time_of_day,
            },
        }

    def _build_text_representation(self, name: str, category: str, data: dict) -> str:
        """Build a rich text representation of a place for embedding."""
        parts = [name, category]

        description = data.get("description", data.get("overview", ""))
        if description:
            parts.append(description)

        features = data.get("features", data.get("amenities", data.get("highlights", [])))
        if isinstance(features, list):
            parts.append(" ".join(str(f) for f in features))
        elif isinstance(features, str):
            parts.append(features)

        reviews = data.get("reviews", [])
        if reviews:
            review_texts = []
            for r in reviews[:10]:
                if isinstance(r, dict):
                    review_texts.append(r.get("text", r.get("content", r.get("review", ""))))
                elif isinstance(r, str):
                    review_texts.append(r)
            parts.append(" ".join(review_texts))

        tags = data.get("tags", data.get("keywords", []))
        if isinstance(tags, list):
            parts.append(" ".join(str(t) for t in tags))

        location = data.get("location", data.get("address", data.get("area", "")))
        if location:
            if isinstance(location, dict):
                parts.append(str(location.get("area", "")))
                parts.append(str(location.get("city", "")))
            else:
                parts.append(str(location))

        price = data.get("price_range", data.get("price_level", ""))
        if price:
            parts.append(str(price))

        rating = data.get("rating", "")
        if rating:
            parts.append(f"rated {rating}")

        return " ".join(str(p) for p in parts if p)

    def _analyze_query(self, query: str) -> dict:
        """Analyze the user query to extract intent and category hints."""
        query_lower = query.lower()

        category_keywords = {
            "restaurants": ["restaurant", "food", "eat", "dine", "cafe", "bistro", "cuisine", "meal"],
            "beaches": ["beach", "shore", "ocean", "sea", "swim", "coast", "sand", "waterfront"],
            "temples": ["temple", "worship", "prayer", "spiritual", "sacred", "religious", "shrine"],
            "malls": ["mall", "shopping", "store", "shop", "market", "retail", "buy"],
        }

        detected_categories = []
        for cat, keywords in category_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected_categories.append(cat)

        time_hints = {
            "morning": ["morning", "breakfast", "early", "sunrise"],
            "afternoon": ["afternoon", "lunch", "midday"],
            "evening": ["evening", "dinner", "sunset", "night"],
            "night": ["night", "late", "midnight", "party"],
        }

        detected_time = None
        for time_label, keywords in time_hints.items():
            if any(kw in query_lower for kw in keywords):
                detected_time = time_label
                break

        return {
            "original_query": query,
            "detected_categories": detected_categories,
            "detected_time": detected_time,
            "query_length": len(query),
        }

    def _extract_highlights(self, doc: PlaceDocument, query: str) -> list[str]:
        """Extract key highlights from a document relevant to the query."""
        highlights = []
        text = doc.text_representation.lower()
        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))

        for word in query_words:
            if word in text:
                idx = text.find(word)
                start = max(0, idx - 30)
                end = min(len(text), idx + len(word) + 30)
                snippet = doc.text_representation[start:end].strip()
                if snippet and snippet not in highlights:
                    highlights.append(snippet)

        rating = doc.raw_data.get("rating")
        if rating:
            highlights.append(f"Rating: {rating}")

        price = doc.raw_data.get("price_range")
        if price:
            highlights.append(f"Price: {price}")

        return highlights[:5]

    def get_stats(self) -> dict:
        """Return pipeline statistics."""
        category_counts = {}
        for doc in self.documents:
            category_counts[doc.category] = category_counts.get(doc.category, 0) + 1

        return {
            "total_documents": len(self.documents),
            "categories": category_counts,
            "is_fitted": self.is_fitted,
            "vectorizer_features": (
                self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0
            ),
        }

    def clear(self):
        """Clear all stored documents and reset the pipeline."""
        self.documents = []
        self.tfidf_matrix = None
        self.is_fitted = False
        self._category_indices = {}
        logger.info("RAG pipeline cleared")
