"""
API Client for Water Meter Reading API
"""

import os
import requests
from typing import Dict, Any, Optional
from src.api_testing_ui.config import (
    API_BASE_URL,
    API_TIMEOUT,
    API_ENDPOINT
)


class APIError(Exception):
    """Custom exception for API errors"""
    def __init__(self, message: str, code: str = "UNKNOWN", details: Optional[Dict] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


class MeterReadingAPIClient:
    """Client for Water Meter Reading API"""

    def __init__(self, base_url: str = API_BASE_URL, timeout: int = API_TIMEOUT):
        """
        Initialize API client

        Args:
            base_url: API base URL (default: from config)
            timeout: Request timeout in seconds (default: from config)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.endpoint = API_ENDPOINT

    def test_reading(
        self,
        image_path: str,
        meter_id: str = "TEST-001",
        gps: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Send test reading request to API

        Args:
            image_path: Path to water meter image file
            meter_id: Optional meter identifier
            gps: Optional GPS coordinates {"latitude": float, "longitude": float}

        Returns:
            API response dictionary with reading data

        Raises:
            APIError: If API request fails
        """
        # Check if file exists first
        if not os.path.exists(image_path):
            raise APIError(
                f"Image file not found: {image_path}",
                code="FILE_NOT_FOUND"
            )

        url = f"{self.base_url}{self.endpoint}"

        # Prepare request payload
        data = {"meter_id": meter_id}

        if gps:
            if "latitude" in gps:
                data["latitude"] = gps["latitude"]
            if "longitude" in gps:
                data["longitude"] = gps["longitude"]

        try:
            with open(image_path, 'rb') as image_file:
                files = {"image": (image_path, image_file, "image/jpeg")}

                response = requests.post(
                    url,
                    files=files,
                    data=data,
                    timeout=self.timeout
                )

        except requests.exceptions.Timeout:
            raise APIError(
                "API request timed out",
                code="TIMEOUT",
                details={"timeout_seconds": self.timeout}
            )
        except requests.exceptions.ConnectionError as e:
            raise APIError(
                "Cannot connect to API server",
                code="CONNECTION_ERROR",
                details={"url": url, "error": str(e)}
            )
        except APIError:
            # Re-raise APIError (e.g., FILE_NOT_FOUND) without modification
            raise
        except Exception as e:
            raise APIError(
                f"Unexpected error: {str(e)}",
                code="UNKNOWN_ERROR"
            )

        return self._validate_response(response)

    def _validate_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Validate and parse API response

        Args:
            response: HTTP response object

        Returns:
            Parsed JSON response dictionary

        Raises:
            APIError: If response indicates error
        """
        try:
            data = response.json()
        except ValueError:
            raise APIError(
                "Invalid JSON response from API",
                code="INVALID_JSON",
                details={"status_code": response.status_code}
            )

        if response.status_code == 200:
            if data.get("success"):
                return data
            else:
                error_info = data.get("error", {})
                raise APIError(
                    error_info.get("message", "Unknown API error"),
                    code=error_info.get("code", "API_ERROR"),
                    details=error_info.get("details", {})
                )

        # Handle error status codes
        error_messages = {
            400: "Bad Request - Invalid parameters",
            404: "API endpoint not found",
            500: "Internal Server Error"
        }

        message = error_messages.get(
            response.status_code,
            f"HTTP {response.status_code} error"
        )

        raise APIError(
            message,
            code=f"HTTP_{response.status_code}",
            details={"status_code": response.status_code, "response": data}
        )
