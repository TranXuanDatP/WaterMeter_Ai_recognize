"""
Unit tests for Streamlit app helper functions
"""

import pytest
from src.api_testing_ui.app import (
    get_confidence_color,
    get_routing_status
)


class TestAppHelpers:
    """Test app helper functions"""

    def test_get_confidence_color_green(self):
        """Test green color for high confidence"""
        result = get_confidence_color(0.95)
        assert result == (34, 197, 94)  # GREEN

    def test_get_confidence_color_yellow(self):
        """Test yellow color for medium confidence"""
        result = get_confidence_color(0.80)
        assert result == (234, 179, 8)  # YELLOW

        # Test lower bound
        result = get_confidence_color(0.70)
        assert result == (234, 179, 8)  # YELLOW

    def test_get_confidence_color_red(self):
        """Test red color for low confidence"""
        result = get_confidence_color(0.50)
        assert result == (239, 68, 68)  # RED

        # Test upper bound
        result = get_confidence_color(0.69)
        assert result == (239, 68, 68)  # RED

    def test_get_routing_status_auto_verified(self):
        """Test auto-verified status"""
        result = get_routing_status(0.95)
        assert "Auto-verified" in result
        assert "✅" in result

    def test_get_routing_status_suggest_review(self):
        """Test suggest review status"""
        result = get_routing_status(0.80)
        assert "Suggest" in result
        assert "⚠️" in result

        # Test lower bound
        result = get_routing_status(0.70)
        assert "Suggest" in result

    def test_get_routing_status_manual_review(self):
        """Test manual review status"""
        result = get_routing_status(0.50)
        assert "Manual Review" in result
        assert "🔴" in result

        # Test upper bound
        result = get_routing_status(0.69)
        assert "Manual Review" in result
