"""
Unit tests for main Streamlit app
"""

import pytest
from unittest.mock import patch, MagicMock
from src.api_testing_ui.app import (
    set_page_config,
    render_header,
    get_confidence_color,
    get_routing_status
)


class TestApp:
    """Test Streamlit app functions"""

    @patch('src.api_testing_ui.app.st.set_page_config')
    def test_set_page_config(self, mock_set_page_config):
        """Test page configuration is set"""
        set_page_config()

        mock_set_page_config.assert_called_once()
        call_kwargs = mock_set_page_config.call_args[1]
        assert "Water Meter AI" in call_kwargs['page_title']
        assert call_kwargs['layout'] == 'wide'
        assert call_kwargs['page_icon'] == "🖼️"
        assert call_kwargs['initial_sidebar_state'] == "expanded"

    @patch('src.api_testing_ui.app.st.title')
    @patch('src.api_testing_ui.app.st.markdown')
    def test_render_header(self, mock_markdown, mock_title):
        """Test header rendering"""
        render_header()

        mock_title.assert_called_once()
        title_arg = mock_title.call_args[0][0]
        assert "Water Meter AI" in title_arg

    def test_get_confidence_color_green(self):
        """Test green color for high confidence (>0.90)"""
        result = get_confidence_color(0.95)
        assert result == (34, 197, 94)  # GREEN

        # Test boundary
        result = get_confidence_color(0.91)
        assert result == (34, 197, 94)

    def test_get_confidence_color_yellow(self):
        """Test yellow color for medium confidence (0.70-0.90)"""
        result = get_confidence_color(0.80)
        assert result == (234, 179, 8)  # YELLOW

        # Test lower bound
        result = get_confidence_color(0.70)
        assert result == (234, 179, 8)

        # Test upper bound
        result = get_confidence_color(0.90)
        assert result == (234, 179, 8)

    def test_get_confidence_color_red(self):
        """Test red color for low confidence (<0.70)"""
        result = get_confidence_color(0.50)
        assert result == (239, 68, 68)  # RED

        # Test boundary
        result = get_confidence_color(0.69)
        assert result == (239, 68, 68)

    def test_get_routing_status_auto_verified(self):
        """Test auto-verified status for high confidence"""
        result = get_routing_status(0.95)
        assert "Auto-verified" in result
        assert "✅" in result

    def test_get_routing_status_suggest_review(self):
        """Test suggest review status for medium confidence"""
        result = get_routing_status(0.80)
        assert "Suggest" in result
        assert "⚠️" in result

        # Test lower bound
        result = get_routing_status(0.70)
        assert "Suggest" in result

    def test_get_routing_status_manual_review(self):
        """Test manual review status for low confidence"""
        result = get_routing_status(0.50)
        assert "Manual Review" in result
        assert "🔴" in result

        # Test upper bound
        result = get_routing_status(0.69)
        assert "Manual Review" in result
