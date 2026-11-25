import sys
from unittest.mock import patch
from pathlib import Path

import pytest


def test_tracker_imports_mlflow():
    """Verify tracker can import mlflow successfully."""
    import tracker

    assert hasattr(tracker, "MLFLOW_AVAILABLE")
    assert hasattr(tracker, "is_available")


def test_config_works():
    """Verify config loads fine."""
    from config import Settings

    s = Settings()
    assert s.chunk_size == 1000


def test_tracker_available_with_mlflow():
    """Verify tracker reports available when mlflow is installed."""
    import tracker

    assert tracker.is_available() is True
