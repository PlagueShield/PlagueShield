"""Few-shot resistance inference, uncertainty estimation and abstention."""

from .exemplars import EXEMPLAR_BANK, extract_features, few_shot_vote
from .ood import assess as assess_ood
from .prior import RESISTANCE_PRIOR, ResistancePosterior

__all__ = [
    "EXEMPLAR_BANK",
    "extract_features",
    "few_shot_vote",
    "assess_ood",
    "RESISTANCE_PRIOR",
    "ResistancePosterior",
]
