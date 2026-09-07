#!/usr/bin/env python3
"""Reproduce the numerical tables used in Chapter 1."""

from __future__ import annotations

import math

from ch01_minimal_inference import dot, softmax


def temperature_rows() -> list[tuple[float, list[float]]]:
    logits = [4.0, 2.0, 1.0, 0.0]
    return [
        (temperature, softmax([value / temperature for value in logits]))
        for temperature in (0.5, 1.0, 2.0)
    ]


def attention_walkthrough() -> tuple[list[float], list[float], list[float]]:
    query = [1.0, 0.5]
    keys = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    values = [[1.0, 0.0], [0.0, 2.0], [2.0, 1.0]]
    scores = [dot(query, key) / math.sqrt(2.0) for key in keys]
    probabilities = softmax(scores)
    output = [
        sum(probability * value[dimension] for probability, value in zip(probabilities, values))
        for dimension in range(2)
    ]
    return scores, probabilities, output


def format_row(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def main() -> None:
    print("temperature distributions for logits [4, 2, 1, 0]")
    for temperature, probabilities in temperature_rows():
        print(f"T={temperature:.1f}: {format_row(probabilities)}")

    scores, probabilities, output = attention_walkthrough()
    print("attention scores:", format_row(scores))
    print("attention probabilities:", format_row(probabilities))
    print("attention output:", format_row(output))


if __name__ == "__main__":
    main()
