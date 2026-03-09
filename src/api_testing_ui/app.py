"""
Main Streamlit Application for API Testing UI
Complete implementation with all features
"""

import streamlit as st
import os
from PIL import Image
from io import BytesIO
import json
from datetime import datetime

from src.api_testing_ui.config import (
    PAGE_TITLE,
    PAGE_LAYOUT,
    API_BASE_URL,
    API_ENDPOINT,
    SUPPORTED_IMAGE_TYPES,
    MAX_IMAGE_SIZE_MB,
    MAX_HISTORY_SIZE
)
from src.api_testing_ui.api_client import MeterReadingAPIClient, APIError


def set_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout=PAGE_LAYOUT,
        page_icon="🖼️",
        initial_sidebar_state="expanded"
    )


def render_header():
    """Render application header"""
    st.title("🖼️ Water Meter AI - Testing UI")
    st.markdown("---")
    st.markdown("""
    **Test the Water Meter Reading API** with image upload and real-time results display.

    **API Endpoint:** `{api_base_url}{api_endpoint}`
    """.format(api_base_url=API_BASE_URL, api_endpoint=API_ENDPOINT))
    st.markdown("---")


def get_confidence_color(confidence: float) -> tuple:
    """
    Get color based on confidence score

    Args:
        confidence: Confidence score (0.0 - 1.0)

    Returns:
        RGB color tuple
    """
    if confidence > 0.90:
        return (34, 197, 94)  # GREEN
    elif confidence >= 0.70:
        return (234, 179, 8)  # YELLOW
    else:
        return (239, 68, 68)  # RED


def get_routing_status(confidence: float) -> str:
    """
    Get routing status based on confidence

    Args:
        confidence: Confidence score (0.0 - 1.0)

    Returns:
        Status string
    """
    if confidence > 0.90:
        return "✅ Auto-verified"
    elif confidence >= 0.70:
        return "⚠️ Suggest Manual Review"
    else:
        return "🔴 Manual Review Required"


def render_input_section():
    """Render input section with image upload and parameters"""
    st.subheader("📤 Upload Image & Parameters")

    # Image upload
    uploaded_file = st.file_uploader(
        "Upload water meter image",
        type=SUPPORTED_IMAGE_TYPES,
        help="Upload JPEG or PNG image (max 10MB)",
        key="image_uploader"
    )

    # Display image preview and metadata
    if uploaded_file:
        col1, col2 = st.columns(2)

        with col1:
            try:
                image = Image.open(uploaded_file)
                # Resize for preview (max 400x400)
                image.thumbnail((400, 400))
                st.image(image, caption="Preview", use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")

        with col2:
            st.markdown("**Image Metadata**")
            st.text(f"Name: {uploaded_file.name}")
            st.text(f"Size: {uploaded_file.size / (1024*1024):.2f} MB")

            try:
                img = Image.open(uploaded_file)
                st.text(f"Dimensions: {img.width} × {img.height} px")
                st.text(f"Type: {uploaded_file.type}")
            except:
                pass

        if st.button("🗑️ Clear Image", key="clear_image"):
            st.rerun()

    st.markdown("---")

    # API Parameters
    st.markdown("**API Parameters**")

    col1, col2 = st.columns(2)
    with col1:
        meter_id = st.text_input(
            "Meter ID",
            value="TEST-001",
            help="Optional meter identifier",
            key="meter_id"
        )

    with col2:
        debug_mode = st.checkbox(
            "Include debug info",
            value=False,
            help="Show detailed debug information in response"
        )

    # GPS coordinates
    with st.expander("📍 GPS Coordinates (Optional)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input(
                "Latitude",
                value=None,
                placeholder="e.g., 21.0285",
                format="%.4f",
                key="latitude"
            )
        with col2:
            longitude = st.number_input(
                "Longitude",
                value=None,
                placeholder="e.g., 105.8542",
                format="%.4f",
                key="longitude"
            )

    # Submit button
    submit_disabled = uploaded_file is None
    submit_button = st.button(
        "🚀 Test API",
        type="primary",
        disabled=submit_disabled,
        use_container_width=True
    )

    return uploaded_file, meter_id, latitude, longitude, debug_mode, submit_button


def render_results_section(response_data: dict, processing_time: float = None):
    """
    Render results display section

    Args:
        response_data: API response data
        processing_time: Processing time in milliseconds
    """
    st.subheader("📥 Results")

    if not response_data.get("success"):
        # Error response
        error = response_data.get("error", {})
        st.error(f"**Error:** {error.get('message', 'Unknown error')}")
        st.code(f"Error Code: {error.get('code', 'UNKNOWN')}", language="bash")

        if error.get("details"):
            with st.expander("🔍 Error Details"):
                st.json(error.get("details"))
        return

    # Success response
    reading = response_data.get("reading", {})
    value = reading.get("value", "N/A")
    confidence = reading.get("confidence", 0.0)

    # Color-coded confidence box
    color = get_confidence_color(confidence)
    color_hex = "#%02x%02x%02x" % color

    # Main result display
    st.markdown(f"""
    <div style="background-color: {color_hex}; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">{value}</h1>
    </div>
    """, unsafe_allow_html=True)

    # Confidence and routing status
    col1, col2, col3 = st.columns(3)

    with col1:
        routing_status = get_routing_status(confidence)
        st.markdown(f"**Routing:** {routing_status}")

    with col2:
        confidence_pct = confidence * 100
        st.markdown(f"**Confidence:** {confidence_pct:.1f}%")

    with col3:
        if processing_time:
            st.markdown(f"**Time:** {processing_time:.0f} ms")

    # Confidence progress bar
    st.progress(confidence)

    st.markdown("---")

    # Digit breakdown
    if "digits" in reading:
        with st.expander("🔢 Digit Details", expanded=True):
            digits = reading["digits"]
            digit_probs = reading.get("digit_probabilities", [])

            for i, digit in enumerate(digits):
                prob = digit_probs[i][digit] if i < len(digit_probs) else 0.0
                st.markdown(f"**Digit {i+1}:** {digit} ({prob*100:.1f}% confidence)")

    # Pointer breakdown
    if "pointers" in reading:
        with st.expander("🎯 Pointer Details", expanded=False):
            pointers = reading["pointers"]
            pointer_value = reading.get("pointer_value", 0.0)

            st.markdown(f"**Total Pointer Value:** {pointer_value:.2f}")

            for i, ptr in enumerate(pointers):
                st.markdown(f"""
                **Pointer {i+1}:**
                - Value: {ptr['value']}
                - Angle: {ptr['angle']:.1f}°
                - Scale: ×{ptr['scale']}
                """)

    # Debug panel
    if st.session_state.get("debug_mode", False):
        with st.expander("🐛 Debug Info", expanded=False):
            st.markdown("**Full API Response:**")
            st.json(response_data)

            if "processing_time_ms" in response_data:
                st.markdown(f"**Processing Time:** {response_data['processing_time_ms']:.0f} ms")

            if "pipeline_results" in response_data:
                st.markdown("**Pipeline Results:**")
                st.json(response_data["pipeline_results"])


def render_error_section(error: APIError):
    """
    Render error display section

    Args:
        error: APIError exception
    """
    st.subheader("❌ Error")

    # Error message with icon
    error_icons = {
        "CONNECTION_ERROR": "🔌",
        "TIMEOUT": "⏱️",
        "FILE_NOT_FOUND": "📁",
        "HTTP_400": "⚠️",
        "HTTP_404": "🔍",
        "HTTP_500": "💥",
        "NO_METER_DETECTED": "🚷",
    }

    icon = error_icons.get(error.code, "❌")
    st.error(f"{icon} **{error.message}**")

    # Error code
    st.code(f"Error Code: {error.code}", language="bash")

    # Suggestions based on error type
    suggestions = {
        "CONNECTION_ERROR": [
            "Check if the API server is running",
            "Verify the API base URL in config.py",
            "Check network connection"
        ],
        "TIMEOUT": [
            f"Request exceeded {API_BASE_URL} timeout",
            "Try with a smaller image",
            "Check if API server is overloaded"
        ],
        "FILE_NOT_FOUND": [
            "Verify the image file path is correct",
            "Check if the file exists",
            "Try re-uploading the image"
        ],
        "NO_METER_DETECTED": [
            "The AI could not detect a water meter in the image",
            "Try a clearer image with better lighting",
            "Ensure the meter is fully visible in the frame"
        ],
        "HTTP_400": "Bad Request - Check API parameters",
        "HTTP_404": "API endpoint not found - Check API server configuration",
        "HTTP_500": "Internal Server Error - Contact API support",
    }

    if error.code in suggestions:
        st.markdown("**Suggestions:**")
        for suggestion in suggestions[error.code]:
            st.markdown(f"• {suggestion}")

    # Error details
    if error.details:
        with st.expander("🔍 Error Details"):
            st.json(error.details)

    # Retry button
    if st.button("🔄 Retry", key="retry_button"):
        st.rerun()


def render_session_history():
    """Render session history in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📜 Recent Tests")

    if "test_history" not in st.session_state:
        st.session_state.test_history = []

    history = st.session_state.test_history

    if not history:
        st.sidebar.info("No tests yet")
        return

    # Display history in reverse order (newest first)
    for i, item in enumerate(reversed(history)):
        timestamp = item.get("timestamp", "")
        value = item.get("value", "N/A")
        confidence = item.get("confidence", 0.0)
        status_icon = "✅" if confidence > 0.90 else ("⚠️" if confidence >= 0.70 else "🔴")

        if st.sidebar.button(
            f"{status_icon} {value} ({confidence*100:.0f}%)",
            key=f"history_{i}",
            use_container_width=True
        ):
            # Restore this result
            st.session_state.restored_result = item
            st.rerun()

    # Clear history button
    if st.sidebar.button("🗑️ Clear History", use_container_width=True):
        st.session_state.test_history = []
        st.rerun()


def add_to_history(response_data: dict):
    """
    Add test result to session history

    Args:
        response_data: API response data
    """
    if "test_history" not in st.session_state:
        st.session_state.test_history = []

    reading = response_data.get("reading", {})
    history_item = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "value": reading.get("value", "N/A"),
        "confidence": reading.get("confidence", 0.0),
        "response": response_data
    }

    st.session_state.test_history.append(history_item)

    # Keep only the most recent MAX_HISTORY_SIZE items
    if len(st.session_state.test_history) > MAX_HISTORY_SIZE:
        st.session_state.test_history = st.session_state.test_history[-MAX_HISTORY_SIZE:]


def main():
    """Main application entry point"""
    set_page_config()
    render_header()

    # Initialize session state
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    # Two-column layout
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file, meter_id, latitude, longitude, debug_mode, submit_button = render_input_section()
        st.session_state.debug_mode = debug_mode

        # Process API request
        if submit_button and uploaded_file:
            with st.spinner("Processing..."):
                try:
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # Prepare GPS data
                    gps = None
                    if latitude is not None and longitude is not None:
                        gps = {"latitude": latitude, "longitude": longitude}

                    # Call API
                    client = MeterReadingAPIClient()
                    result = client.test_reading(temp_path, meter_id, gps)

                    # Store result and add to history
                    st.session_state.last_result = result
                    add_to_history(result)

                    # Clean up temp file
                    os.remove(temp_path)

                    st.success("✅ API call successful!")
                    st.rerun()

                except APIError as e:
                    st.session_state.last_error = e
                    st.rerun()
                except Exception as e:
                    st.session_state.last_error = APIError(f"Unexpected error: {str(e)}", "UNKNOWN_ERROR")
                    st.rerun()

    with col2:
        # Display results or errors
        if "last_error" in st.session_state:
            render_error_section(st.session_state.last_error)

        elif st.session_state.last_result:
            render_results_section(st.session_state.last_result)

        elif "restored_result" in st.session_state:
            result = st.session_state.restored_result["response"]
            render_results_section(result)

        else:
            st.info("👆 Upload an image and click 'Test API' to see results")

    # Render session history in sidebar
    render_session_history()


if __name__ == "__main__":
    main()
