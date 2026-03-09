"""
Unit tests for API Client
"""

import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import requests
from src.api_testing_ui.api_client import MeterReadingAPIClient, APIError


class TestMeterReadingAPIClient:
    """Test Meter Reading API Client"""

    def test_init_default_params(self):
        """Test client initialization with defaults"""
        client = MeterReadingAPIClient()

        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 30
        assert client.endpoint == "/api/v1/read-meter"

    def test_init_custom_params(self):
        """Test client initialization with custom params"""
        client = MeterReadingAPIClient(base_url="http://example.com", timeout=60)

        assert client.base_url == "http://example.com"
        assert client.timeout == 60

    @patch('src.api_testing_ui.api_client.os.path.exists', return_value=True)
    @patch('src.api_testing_ui.api_client.open', create=True)
    @patch('src.api_testing_ui.api_client.requests.post')
    def test_successful_reading(self, mock_post, mock_open, mock_exists):
        """Test successful API response parsing"""
        # Mock the file object
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.read.return_value = b'image_data'
        mock_open.return_value = mock_file

        # Use SimpleNamespace for response with actual values
        mock_response = SimpleNamespace(
            status_code=200,
            json=lambda: {"success": True, "reading": {"value": "12345", "confidence": 0.95}}
        )
        mock_post.return_value = mock_response

        client = MeterReadingAPIClient()
        result = client.test_reading("/path/to/image.jpg")

        assert result["success"] is True
        assert result["reading"]["value"] == "12345"
        mock_post.assert_called_once()

    @patch('src.api_testing_ui.api_client.os.path.exists', return_value=True)
    @patch('src.api_testing_ui.api_client.open', create=True)
    @patch('src.api_testing_ui.api_client.requests.post')
    def test_connection_error(self, mock_post, mock_open, mock_exists):
        """Test connection error handling"""
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_file

        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        client = MeterReadingAPIClient()
        with pytest.raises(APIError) as exc_info:
            client.test_reading("/path/to/image.jpg")

        assert exc_info.value.code == "CONNECTION_ERROR"
        assert "Cannot connect" in exc_info.value.message

    @patch('src.api_testing_ui.api_client.os.path.exists', return_value=True)
    @patch('src.api_testing_ui.api_client.open', create=True)
    @patch('src.api_testing_ui.api_client.requests.post')
    def test_timeout_error(self, mock_post, mock_open, mock_exists):
        """Test timeout error handling"""
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_file

        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        client = MeterReadingAPIClient()
        with pytest.raises(APIError) as exc_info:
            client.test_reading("/path/to/image.jpg")

        assert exc_info.value.code == "TIMEOUT"
        assert "timed out" in exc_info.value.message

    @patch('src.api_testing_ui.api_client.os.path.exists', return_value=False)
    def test_file_not_found_error(self, mock_exists):
        """Test file not found error"""
        client = MeterReadingAPIClient()

        with pytest.raises(APIError) as exc_info:
            client.test_reading("/nonexistent/image.jpg")

        assert exc_info.value.code == "FILE_NOT_FOUND"

    @patch('src.api_testing_ui.api_client.os.path.exists', return_value=True)
    @patch('src.api_testing_ui.api_client.open', create=True)
    @patch('src.api_testing_ui.api_client.requests.post')
    def test_api_error_response(self, mock_post, mock_open, mock_exists):
        """Test API returns error response"""
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_file

        mock_response = SimpleNamespace(
            status_code=200,
            json=lambda: {
                "success": False,
                "error": {
                    "code": "NO_METER_DETECTED",
                    "message": "No water meter detected",
                    "details": {"confidence": 0.32}
                }
            }
        )
        mock_post.return_value = mock_response

        client = MeterReadingAPIClient()
        with pytest.raises(APIError) as exc_info:
            client.test_reading("/path/to/image.jpg")

        assert exc_info.value.code == "NO_METER_DETECTED"
        assert "No water meter detected" in exc_info.value.message

    @patch('src.api_testing_ui.api_client.os.path.exists', return_value=True)
    @patch('src.api_testing_ui.api_client.open', create=True)
    @patch('src.api_testing_ui.api_client.requests.post')
    def test_http_400_error(self, mock_post, mock_open, mock_exists):
        """Test HTTP 400 error handling"""
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_file

        mock_response = SimpleNamespace(
            status_code=400,
            json=lambda: {"error": "Bad request"}
        )
        mock_post.return_value = mock_response

        client = MeterReadingAPIClient()
        with pytest.raises(APIError) as exc_info:
            client.test_reading("/path/to/image.jpg")

        assert exc_info.value.code == "HTTP_400"
        assert "Bad Request" in exc_info.value.message

    @patch('src.api_testing_ui.api_client.os.path.exists', return_value=True)
    @patch('src.api_testing_ui.api_client.open', create=True)
    @patch('src.api_testing_ui.api_client.requests.post')
    def test_includes_gps_data(self, mock_post, mock_open, mock_exists):
        """Test GPS coordinates are included in request"""
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_file

        mock_response = SimpleNamespace(
            status_code=200,
            json=lambda: {"success": True}
        )
        mock_post.return_value = mock_response

        client = MeterReadingAPIClient()
        gps = {"latitude": 21.0285, "longitude": 105.8542}
        client.test_reading("/path/to/image.jpg", gps=gps)

        call_kwargs = mock_post.call_args[1]
        assert "data" in call_kwargs
        assert call_kwargs["data"]["latitude"] == 21.0285
        assert call_kwargs["data"]["longitude"] == 105.8542
