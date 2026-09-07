#!/usr/bin/env python3
"""A dependency-free decoder-only inference model for Chapter 1.

The weights are random, so the generated text has no semantic value. The point
is to make causal attention, prefill, cached decode, logits, and sampling small
enough to inspect and test.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


Vector = list[float]
Matrix = list[Vector]
KVCache = tuple[list[Vector], list[Vector]]


def dot(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right))


def add(left: Vector, right: Vector) -> Vector:
    return [a + b for a, b in zip(left, right)]


def matvec(weight: Matrix, vector: Vector) -> Vector:
    return [dot(row, vector) for row in weight]


def scale(vector: Vector, value: float) -> Vector:
    return [item * value for item in vector]


def rms_norm(vector: Vector, eps: float = 1e-6) -> Vector:
    mean_square = sum(item * item for item in vector) / len(vector)
    return scale(vector, 1.0 / math.sqrt(mean_square + eps))


def silu(value: float) -> float:
    return value / (1.0 + math.exp(-value))


def softmax(values: Vector) -> Vector:
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    denominator = sum(exps)
    return [value / denominator for value in exps]


def rotate_pairs(vector: Vector, position: int, base: float = 10_000.0) -> Vector:
    """Apply a small RoPE-style rotation to adjacent dimension pairs."""
    assert len(vector) % 2 == 0
    result = vector[:]
    half_dim = len(vector) // 2
    for pair in range(half_dim):
        left = 2 * pair
        right = left + 1
        frequency = base ** (-pair / max(1, half_dim))
        angle = position * frequency
        cos_value = math.cos(angle)
        sin_value = math.sin(angle)
        result[left] = vector[left] * cos_value - vector[right] * sin_value
        result[right] = vector[left] * sin_value + vector[right] * cos_value
    return result


def random_matrix(rows: int, columns: int, rng: random.Random) -> Matrix:
    bound = 1.0 / math.sqrt(columns)
    return [
        [rng.uniform(-bound, bound) for _ in range(columns)] for _ in range(rows)
    ]


def max_abs_diff(left: Vector, right: Vector) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


def sample_logits(
    logits: Vector,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 0,
) -> int:
    """Sample after temperature, top-k, and top-p filtering."""
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if temperature == 0:
        return max(range(len(logits)), key=logits.__getitem__)

    scaled_logits = [value / temperature for value in logits]
    candidate_ids = sorted(
        range(len(logits)), key=scaled_logits.__getitem__, reverse=True
    )
    all_probs = softmax(scaled_logits)
    nucleus_ids: list[int] = []
    cumulative = 0.0
    for token_id in candidate_ids:
        nucleus_ids.append(token_id)
        probability = all_probs[token_id]
        cumulative += probability
        if cumulative >= top_p:
            break

    max_candidates = top_k if top_k > 0 else len(candidate_ids)
    kept_ids = nucleus_ids[:max_candidates]
    kept_probs = [all_probs[token_id] for token_id in kept_ids]

    total = sum(kept_probs)
    threshold = random.Random(seed).random()
    cumulative = 0.0
    for token_id, probability in zip(kept_ids, kept_probs):
        cumulative += probability / total
        if threshold <= cumulative:
            return token_id
    return kept_ids[-1]


@dataclass(frozen=True)
class ToyConfig:
    vocab_size: int = 8
    hidden_size: int = 8
    intermediate_size: int = 12
    seed: int = 7


class ToyDecoder:
    """One decoder layer, one attention head, tied output embeddings."""

    def __init__(self, config: ToyConfig = ToyConfig()) -> None:
        self.config = config
        rng = random.Random(config.seed)
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        self.embeddings = random_matrix(config.vocab_size, hidden, rng)
        self.wq = random_matrix(hidden, hidden, rng)
        self.wk = random_matrix(hidden, hidden, rng)
        self.wv = random_matrix(hidden, hidden, rng)
        self.wo = random_matrix(hidden, hidden, rng)
        self.w_gate = random_matrix(intermediate, hidden, rng)
        self.w_up = random_matrix(intermediate, hidden, rng)
        self.w_down = random_matrix(hidden, intermediate, rng)

    def _qkv(self, token_id: int, position: int) -> tuple[Vector, Vector, Vector, Vector]:
        residual = self.embeddings[token_id][:]
        normalized = rms_norm(residual)
        query = rotate_pairs(matvec(self.wq, normalized), position)
        key = rotate_pairs(matvec(self.wk, normalized), position)
        value = matvec(self.wv, normalized)
        return residual, query, key, value

    def _finish_layer(
        self, residual: Vector, query: Vector, keys: list[Vector], values: list[Vector]
    ) -> Vector:
        scores = [
            dot(query, key) / math.sqrt(self.config.hidden_size) for key in keys
        ]
        probabilities = softmax(scores)
        context = [0.0] * self.config.hidden_size
        for probability, value in zip(probabilities, values):
            context = add(context, scale(value, probability))

        after_attention = add(residual, matvec(self.wo, context))
        normalized = rms_norm(after_attention)
        gate = matvec(self.w_gate, normalized)
        up = matvec(self.w_up, normalized)
        activated = [silu(left) * right for left, right in zip(gate, up)]
        return add(after_attention, matvec(self.w_down, activated))

    def _logits(self, hidden_state: Vector) -> Vector:
        normalized = rms_norm(hidden_state)
        return [dot(embedding, normalized) for embedding in self.embeddings]

    def forward_full(self, token_ids: list[int]) -> list[Vector]:
        """Recompute the entire causal sequence without reusing a cache."""
        projected = [self._qkv(token_id, position) for position, token_id in enumerate(token_ids)]
        keys = [item[2] for item in projected]
        values = [item[3] for item in projected]
        outputs: list[Vector] = []
        for position, (residual, query, _, _) in enumerate(projected):
            hidden = self._finish_layer(
                residual, query, keys[: position + 1], values[: position + 1]
            )
            outputs.append(self._logits(hidden))
        return outputs

    def prefill(self, token_ids: list[int]) -> tuple[list[Vector], KVCache]:
        """Build K/V once for the prompt and return logits for all positions."""
        projected = [self._qkv(token_id, position) for position, token_id in enumerate(token_ids)]
        keys = [item[2] for item in projected]
        values = [item[3] for item in projected]
        logits: list[Vector] = []
        for position, (residual, query, _, _) in enumerate(projected):
            hidden = self._finish_layer(
                residual, query, keys[: position + 1], values[: position + 1]
            )
            logits.append(self._logits(hidden))
        return logits, (keys, values)

    def decode_one(self, token_id: int, cache: KVCache) -> Vector:
        """Append one token's K/V and attend over the complete cached prefix."""
        keys, values = cache
        position = len(keys)
        residual, query, key, value = self._qkv(token_id, position)
        keys.append(key)
        values.append(value)
        hidden = self._finish_layer(residual, query, keys, values)
        return self._logits(hidden)


def attention_score_counts(prompt_tokens: int, generated_tokens: int) -> tuple[int, int]:
    """Return QK dot-product counts for full recompute and cached generation."""
    no_cache = sum(
        length * (length + 1) // 2
        for length in range(prompt_tokens, prompt_tokens + generated_tokens)
    )
    cached = prompt_tokens * (prompt_tokens + 1) // 2 + sum(
        prompt_tokens + step for step in range(1, generated_tokens)
    )
    return no_cache, cached


def run_demo() -> None:
    vocabulary = ["<bos>", "我", "喜欢", "读", "源码", "学习", "。", "<eos>"]
    model = ToyDecoder()
    token_ids = [0, 1, 2, 3]
    cached_logits, cache = model.prefill(token_ids)
    full_logits = model.forward_full(token_ids)
    assert max_abs_diff(cached_logits[-1], full_logits[-1]) < 1e-12

    print("prompt:", " ".join(vocabulary[token_id] for token_id in token_ids))
    print("prefill cache entries per layer:", len(cache[0]))

    for step in range(3):
        recomputed = model.forward_full(token_ids)[-1]
        difference = max_abs_diff(cached_logits[-1], recomputed)
        assert difference < 1e-12
        next_token = sample_logits(
            cached_logits[-1], temperature=0.8, top_k=4, top_p=0.9, seed=step
        )
        print(
            f"decode step {step + 1}: max cached/full diff={difference:.2e}, "
            f"sampled={next_token}:{vocabulary[next_token]}"
        )
        token_ids.append(next_token)
        cached_logits.append(model.decode_one(next_token, cache))

    no_cache, cached = attention_score_counts(prompt_tokens=128, generated_tokens=32)
    print(f"QK score count example (P=128, G=32): no-cache={no_cache}, cached={cached}")


if __name__ == "__main__":
    run_demo()
