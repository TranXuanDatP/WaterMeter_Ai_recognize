"""
Unit tests for API Testing UI configuration
"""

import pytest
from src.api_testing_ui.config import (
    API_BASE_URL,
    API_TIMEOUT,
    API_ENDPOINT,
    MAX_IMAGE_SIZE_MB,
    SUPPORTED_IMAGE_TYPES,
    MAX_HISTORY_SIZE,
    PAGE_TITLE,
    PAGE_LAYOUT
)


class TestConfig:
    """Test configuration values"""

    def test_api_base_url(self):
        """Test API base URL is configured"""
        assert API_BASE_URL == "http://localhost:8000"

    def test_api_timeout(self):
        """Test API timeout is 30 seconds"""
        assert API_TIMEOUT == 30

    def test_api_endpoint(self):
        """Test API endpoint path"""
        assert API_ENDPOINT == "/api/v1/read-meter"

    def test_max_image_size(self):
        """Test max image size is 10MB"""
        assert MAX_IMAGE_SIZE_MB == 10

    def test_supported_image_types(self):
        """Test supported image types include jpg, jpeg, png"""
        assert "jpg" in SUPPORTED_IMAGE_TYPES
        assert "jpeg" in SUPPORTED_IMAGE_TYPES
        assert "png" in SUPPORTED_IMAGE_TYPES
        assert len(SUPPORTED_IMAGE_TYPES) == 3

    def test_max_history_size(self):
        """Test max history size is 10"""
        assert MAX_HISTORY_SIZE == 10

    def test_page_title(self):
        """Test page title"""
        assert "Water Meter AI" in PAGE_TITLE
        assert "Testing UI" in PAGE_TITLE

    def test_page_layout(self):
        """Test page layout is wide"""
        assert PAGE_LAYOUT == "wide"
