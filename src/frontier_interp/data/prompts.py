
"""Handcrafted prompt suites used for debugging and qualitative diagnostics."""

from __future__ import annotations

from typing import Dict, List


HANDCRAFTED_PROMPTS: Dict[str, List[str]] = {
    "factual": [
        "The capital of France is",
        "The chemical symbol for water is",
        "The largest planet in the solar system is",
        "The word after Monday is",
        "The fastest land animal is the",
        "The currency used in Japan is the",
        "The author of Hamlet was",
        "The first month of the year is",
    ],
    "pattern": [
        "winter spring summer winter spring",
        "red blue green red blue",
        "cat dog bird cat dog",
        "one two three one two",
        "apple orange pear apple orange",
        "A B C A B",
        "music art dance music art",
    ],
    "repeated_span": [
        "The phrase was hello there general hello there",
        "Repeat after me: sun moon star sun moon",
        "The key pattern is oak maple pine oak maple",
        "A list of colors: red green blue red green",
        "Paris London Rome Paris London",
    ],
    "reasoning_like": [
        "In a neural network, attention helps the model",
        "To make tea, you usually need hot",
        "The purpose of a seatbelt is to",
        "A good way to boil an egg is to first",
        "The main purpose of a microscope is to",
    ],
}
