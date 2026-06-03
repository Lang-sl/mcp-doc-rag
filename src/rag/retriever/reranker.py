from __future__ import annotations

from rag.models import SearchResult


def _patch_create_position_ids() -> None:
    """Inject create_position_ids_from_input_ids into transformers for jina-reranker compatibility.

    transformers >= 4.46 removed this function, but the jina-reranker-v2 custom
    model code (loaded via trust_remote_code=True) still imports it from
    transformers.models.xlm_roberta.modeling_xlm_roberta.
    """
    import transformers.models.xlm_roberta.modeling_xlm_roberta as xlm_roberta

    if hasattr(xlm_roberta, "create_position_ids_from_input_ids"):
        return

    import torch

    def _create_position_ids_from_input_ids(
        input_ids, padding_idx, past_key_values_length=0
    ):
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (
            torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
        ) * mask
        return incremental_indices.long() + padding_idx

    xlm_roberta.create_position_ids_from_input_ids = (
        _create_position_ids_from_input_ids
    )


class Reranker:
    """jina-reranker-v2 cross-encoder wrapper with lazy loading.

    Model downloads from HuggingFace only on first call to rerank().
    Includes compatibility patch for transformers >= 4.46.
    """

    def __init__(self, model_name: str, max_length: int = 512):
        """Store config. Model loads lazily on first use."""
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._device = "cpu"

    def _ensure_loaded(self) -> None:
        """Lazy load model and tokenizer from HuggingFace.

        Uses trust_remote_code=True for jina models.
        Auto-detects CUDA, falls back to CPU.
        """
        if self._model is not None:
            return

        # Patch for transformers >= 4.46: create_position_ids_from_input_ids
        # was removed but jina-reranker custom model code still imports it.
        _patch_create_position_ids()

        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def rerank(
        self, query: str, candidates: list[SearchResult]
    ) -> list[SearchResult]:
        """Rescore candidates. Returns candidates sorted by new score."""
        if not candidates:
            return candidates

        self._ensure_loaded()

        import torch

        pairs = [
            (query, c.chunk.embed_text[: self.max_length]) for c in candidates
        ]

        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits

        # Handle both single-dim (batch_size,) and multi-dim (batch_size, 1) output
        if logits.dim() == 2:
            scores = logits.squeeze(-1).cpu().tolist()
        else:
            scores = logits.cpu().tolist()

        # Update each candidate's score
        for candidate, score in zip(candidates, scores):
            candidate.score = float(score)

        # Sort descending by new score
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates
