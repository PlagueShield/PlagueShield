"""PlagueShield — multi-agent decision support for suspected Yersinia pestis cases.

Decision support only. This system does not prescribe, withhold or modify
treatment. Every output requires review by a qualified infectious-disease
clinician or clinical microbiologist.
"""

from .models import CaseAssessment, CaseRecord
from .orchestrator import AssessmentPublisher, Pipeline, assess_case

__version__ = "0.1.0"

__all__ = [
    "CaseAssessment",
    "CaseRecord",
    "Pipeline",
    "AssessmentPublisher",
    "assess_case",
    "__version__",
]
