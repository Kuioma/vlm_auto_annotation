"""Evaluator-specific failures with stable CLI semantics."""


class EvaluationError(ValueError):
    """Raised when an evaluation cannot safely produce a result."""

