"""
Confidence Scoring Engine
--------------------------
Combines three signals into a final confidence score scaled to 1–10.

  Signal 1 — ML model probability      (60%)
  Signal 2 — Keyword text match        (30%)
               → short_description carries 2x the weight of full text
                 because it is always filled and captures the core issue
  Signal 3 — Knowledge article check   (10%)

Score > 7  →  auto-assign
Score ≤ 7  →  add work note, route to manual triage
"""

import logging
import re
from typing import Optional

# Import keyword sets from historical_data_agent (single source of truth)
from agents.historical_data_agent import (
    CLOUD_KEYWORDS,
    NETWORK_KEYWORDS,
    APPLICATION_KEYWORDS,
)

logger = logging.getLogger(__name__)

# Map group name → its keyword set
GROUP_KEYWORDS: dict[str, set] = {
    "Cloud Operations":    CLOUD_KEYWORDS,
    "Network Support":     NETWORK_KEYWORDS,
    "Application Support": APPLICATION_KEYWORDS,
}


def _text_confidence(ticket: dict, predicted_group: str) -> float:
    """
    Keyword match score (0.0–1.0) for the predicted group.

    short_description is scored separately and given 2× weight
    because categories/description can be empty or misleading.

    Formula:
        score = (2 × short_desc_hits + full_text_hits) / (3 × saturation)
    Saturates at 3 keyword hits per field.
    """
    short_desc  = (ticket.get("short_description", "") or "").lower()
    description = (ticket.get("description", "")      or "").lower()
    category    = (ticket.get("category", "")         or "").lower()
    subcategory = (ticket.get("subcategory", "")      or "").lower()

    full_text = f"{short_desc} {description} {category} {subcategory}"

    keywords = GROUP_KEYWORDS.get(predicted_group, set())
    if not keywords:
        return 0.5   # neutral for unknown group

    SATURATION = 3  # hits needed to reach max score per field

    sd_hits   = min(sum(1 for kw in keywords if kw in short_desc), SATURATION)
    full_hits = min(sum(1 for kw in keywords if kw in full_text), SATURATION)

    # short_desc weighted 2×, full_text 1×  →  denominator = 3 × SATURATION
    score = (2 * sd_hits + full_hits) / (3 * SATURATION)
    return min(score, 1.0)


def _knowledge_score(predicted_group: str, active_groups: list) -> float:
    return 1.0 if predicted_group in active_groups else 0.0


class ConfidenceScoringEngine:
    """
    Weights:
        ML probability      60%
        Text keyword match  30%   (short_desc 2× weighted inside this signal)
        Knowledge check     10%
    """

    def __init__(
        self,
        historical_weight: float = 0.60,
        text_weight:       float = 0.30,
        knowledge_weight:  float = 0.10,
    ):
        assert abs(historical_weight + text_weight + knowledge_weight - 1.0) < 1e-9
        self.hw = historical_weight
        self.tw = text_weight
        self.kw = knowledge_weight

    def calculate(
        self,
        raw_probability: float,
        ticket: dict,
        predicted_group: str,
        active_groups: list,
    ) -> float:
        text_score      = _text_confidence(ticket, predicted_group)
        knowledge_score = _knowledge_score(predicted_group, active_groups)

        composite = (
            self.hw * raw_probability
            + self.tw * text_score
            + self.kw * knowledge_score
        )

        scaled = round(1.0 + composite * 9.0, 1)
        scaled = max(1.0, min(10.0, scaled))

        logger.debug(
            f"Confidence | group='{predicted_group}' "
            f"prob={raw_probability:.2f} text={text_score:.2f} "
            f"knowledge={knowledge_score:.2f} → {scaled}/10"
        )
        return scaled
