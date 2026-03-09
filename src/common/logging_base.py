"""
Base Logging Framework for Water Meter AI Pipeline

Cung cấp decorator base cho từng module M1-M5:
- Dev chỉ viết logic trong function
- Logging tự động (input, output, performance, errors)
- Bật/tắt logging qua biến ENABLE_LOG hoặc .env
- Hỗ trợ DEBUG level cho loop iterations (tránh log spam)

Features:
- A. Không lồng hàm (No nested functions) - Import từ module riêng
- B. DEBUG level cho loop - Tránh log spam trong batch processing
- C. Sanitize numpy arrays - Chỉ log shape/dtype, không log pixel data

Usage:
    from src.common.logging_base import m1_base, m2_base, m3_base, m4_base, m5_base

    @m1_base()
    def detect_watermeter(image):
        # Dev chỉ viết logic, KHÔNG viết log
        return yolo_model(image)
"""

import logging
import time
import os
from functools import wraps
from typing import Any, Callable, Optional, Union
from pathlib import Path

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Biến điều khiển logging: có thể override bằng .env hoặc environment variable
ENABLE_LOG = os.getenv("WATERMETER_ENABLE_LOG", "True").lower() == "true"

# Log level configuration (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = os.getenv("WATERMETER_LOG_LEVEL", "INFO").upper()

# Logging format
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Configure root logger with dynamic level
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
)

logger = logging.getLogger("WaterMeterAI")


# =============================================================================
# BASE DECORATOR FACTORY
# =============================================================================

def create_module_decorator(
    module_name: str,
    enable_log: bool = ENABLE_LOG,
    log_input: bool = True,
    log_output: bool = True,
    log_performance: bool = True,
    log_error: bool = True,
    log_level: str = "INFO",
    use_debug_for_large_inputs: bool = True,
):
    """
    Factory function tạo decorator cho từng module M1-M5.

    Args:
        module_name: Tên module (VD: "M1", "M2", ...)
        enable_log: Bật/tắt logging cho module này
        log_input: Log input parameters
        log_output: Log output/result
        log_performance: Log execution time
        log_error: Log errors with traceback
        log_level: Log level to use (DEBUG, INFO, WARNING, ERROR)
        use_debug_for_large_inputs: Use DEBUG level for large inputs (numpy arrays, images)

    Returns:
        Decorator function

    Example:
        m1_base = create_module_decorator("M1")
        m2_base = create_module_decorator("M2", enable_log=False)  # Tắt log M2
        m3_base = create_module_decorator("M3", log_level="DEBUG")  # DEBUG level cho M3
    """

    def decorator(func: Callable) -> Callable:
        # Mark function with module info for inspection
        func._watermeter_module = module_name
        func._watermeter_log_level = log_level

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Skip all logging if disabled
            if not enable_log:
                return func(*args, **kwargs)

            # Create module-specific logger
            module_logger = logging.getLogger(f"WaterMeterAI.{module_name}")

            # Get log level as integer
            log_level_int = getattr(logging, log_level.upper(), logging.INFO)

            # Log function call with input
            func_name = func.__name__

            if log_input:
                # Sanitize input for logging (avoid large images/arrays)
                sanitized_args, has_large_input = _sanitize_args(args, kwargs)

                # Use DEBUG level for large inputs (numpy arrays, images) to avoid console spam
                if has_large_input and use_debug_for_large_inputs:
                    module_logger.debug(f"→ {func_name} | Input: {sanitized_args}")
                else:
                    module_logger.info(f"→ {func_name} | Input: {sanitized_args}")

            # Track performance
            start_time = time.time()

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Calculate execution time
                elapsed_ms = (time.time() - start_time) * 1000

                # Log output and performance
                if log_output:
                    sanitized_result, has_large_output = _sanitize_result(result)

                    # Use DEBUG level for large outputs
                    if has_large_output and use_debug_for_large_inputs:
                        module_logger.debug(
                            f"✓ {func_name} | Output: {sanitized_result} | Time: {elapsed_ms:.2f}ms"
                        )
                    else:
                        module_logger.info(
                            f"✓ {func_name} | Output: {sanitized_result} | Time: {elapsed_ms:.2f}ms"
                        )
                elif log_performance:
                    module_logger.info(f"✓ {func_name} | Time: {elapsed_ms:.2f}ms")

                return result

            except Exception as e:
                # Calculate time before error
                elapsed_ms = (time.time() - start_time) * 1000

                # Log error with details
                if log_error:
                    module_logger.error(
                        f"✗ {func_name} | Error: {type(e).__name__}: {str(e)} | Time: {elapsed_ms:.2f}ms"
                    )
                raise

        return wrapper

    return decorator


# =============================================================================
# MODULE-SPECIFIC BASE DECORATORS
# =============================================================================

# M1: Watermeter Detection (YOLOv8)
m1_base = create_module_decorator(
    module_name="M1",
    enable_log=os.getenv("WATERMETER_LOG_M1", "True").lower() == "true",
    log_input=True,
    log_output=True,
    log_performance=True,
    log_error=True,
)

# M2: Orientation Alignment (CNN sin/cos regressor)
m2_base = create_module_decorator(
    module_name="M2",
    enable_log=os.getenv("WATERMETER_LOG_M2", "True").lower() == "true",
    log_input=True,
    log_output=True,
    log_performance=True,
    log_error=True,
)

# M3: Component Detection (YOLOv8)
m3_base = create_module_decorator(
    module_name="M3",
    enable_log=os.getenv("WATERMETER_LOG_M3", "True").lower() == "true",
    log_input=True,
    log_output=True,
    log_performance=True,
    log_error=True,
)

# M4: CRNN Digit Recognition (ResNet + BiLSTM + CTC)
m4_base = create_module_decorator(
    module_name="M4",
    enable_log=os.getenv("WATERMETER_LOG_M4", "True").lower() == "true",
    log_input=True,
    log_output=True,
    log_performance=True,
    log_error=True,
)

# M5: Pointer Reading (Geometric calculation)
m5_base = create_module_decorator(
    module_name="M5",
    enable_log=os.getenv("WATERMETER_LOG_M5", "True").lower() == "true",
    log_input=True,
    log_output=True,
    log_performance=True,
    log_error=True,
)

# Bayesian Enhancement Layer
bayesian_base = create_module_decorator(
    module_name="Bayesian",
    enable_log=os.getenv("WATERMETER_LOG_BAYESIAN", "True").lower() == "true",
    log_input=True,
    log_output=True,
    log_performance=True,
    log_error=True,
)


# =============================================================================
# BATCH PROCESSING DECORATORS (For loops, avoid log spam)
# =============================================================================

# M1 Batch: Process multiple images
m1_batch = create_batch_decorator(
    module_name="M1",
    enable_log=os.getenv("WATERMETER_LOG_M1", "True").lower() == "true",
)

# M3 Batch: Process multiple component detections
m3_batch = create_batch_decorator(
    module_name="M3",
    enable_log=os.getenv("WATERMETER_LOG_M3", "True").lower() == "true",
)

# M5 Batch: Process multiple pointers
m5_batch = create_batch_decorator(
    module_name="M5",
    enable_log=os.getenv("WATERMETER_LOG_M5", "True").lower() == "true",
)


# =============================================================================
# HELPER FUNCTIONS FOR LOG SANITIZATION
# =============================================================================

def _sanitize_args(args: tuple, kwargs: dict) -> tuple[str, bool]:
    """
    Sanitize function arguments for logging.

    Returns:
        tuple: (sanitized_string, has_large_input)
        - sanitized_string: String representation of args
        - has_large_input: True if contains large objects (numpy arrays, images)

    Truncate large objects (images, arrays) to avoid log spam.
    For numpy arrays, only log shape, dtype, and min/max values (not pixel data).
    """
    sanitized = []
    has_large_input = False

    # Process positional args
    for i, arg in enumerate(args):
        arg_str, is_large = _sanitize_value(arg)
        sanitized.append(f"arg{i}={arg_str}")
        has_large_input = has_large_input or is_large

    # Process keyword args
    for key, value in kwargs.items():
        val_str, is_large = _sanitize_value(value)
        sanitized.append(f"{key}={val_str}")
        has_large_input = has_large_input or is_large

    return ", ".join(sanitized), has_large_input


def _sanitize_value(value: Any) -> tuple[str, bool]:
    """
    Sanitize a single value for logging.

    Returns:
        tuple: (sanitized_string, is_large)
        - sanitized_string: Safe string representation
        - is_large: True if value is large (numpy array, image, etc.)

    For Computer Vision data (numpy arrays):
    - Only log shape, dtype, and min/max values
    - NEVER log actual pixel data (causes log spam)
    """
    # === Handle NumPy arrays (Computer Vision) ===
    if _is_numpy_array(value):
        # Only log shape, dtype, and stats - NEVER the actual pixel data
        shape_info = f"shape={value.shape}"
        dtype_info = f"dtype={value.dtype}"

        # Add min/max for debugging (useful for checking image normalization)
        if value.size > 0:
            min_val = float(value.min()) if value.dtype.kind in 'biufc' else 'N/A'
            max_val = float(value.max()) if value.dtype.kind in 'biufc' else 'N/A'
            stats_info = f"range=[{min_val:.2f}, {max_val:.2f}]"
        else:
            stats_info = "empty"

        result = f"ndarray({shape_info}, {dtype_info}, {stats_info})"
        return result, True  # Mark as large (use DEBUG level)

    # === Handle PyTorch tensors ===
    elif _is_torch_tensor(value):
        shape_info = f"shape={tuple(value.shape)}"
        dtype_info = f"dtype={value.dtype}"

        if value.numel() > 0:
            min_val = float(value.min()) if value.dtype.is_floating_point else value.min().item()
            max_val = float(value.max()) if value.dtype.is_floating_point else value.max().item()
            stats_info = f"range=[{min_val:.2f}, {max_val:.2f}]"
        else:
            stats_info = "empty"

        result = f"tensor({shape_info}, {dtype_info}, {stats_info})"
        return result, True  # Mark as large

    # === Handle PIL Images ===
    elif hasattr(value, "size") and hasattr(value, "mode"):  # PIL Image
        result = f"Image(size={value.size}, mode={value.mode})"
        return result, True  # Mark as large

    # === Handle strings ===
    elif isinstance(value, str):
        if len(value) > 100:
            return f"'{value[:50]}...{value[-20:]}' (len={len(value)})", False
        return f"'{value}'", False

    # === Handle lists/tuples ===
    elif isinstance(value, (list, tuple)):
        if len(value) > 10:
            # Check if contains numpy arrays
            if len(value) > 0 and _is_numpy_array(value[0]):
                return f"{type(value).__name__}(len={len(value)}, element_type=ndarray)", True
            return f"{type(value).__name__}(len={len(value)}, [...])", True
        return str(value), False

    # === Handle dicts ===
    elif isinstance(value, dict):
        if len(value) > 10:
            return f"dict(keys={len(value)}, {{...}})", True
        return str(value), False

    # === Handle other types ===
    else:
        return str(value), False


def _is_numpy_array(value: Any) -> bool:
    """Check if value is a numpy array."""
    # More robust check that works even if numpy is not imported
    return hasattr(value, 'shape') and hasattr(value, 'dtype') and hasattr(value, 'ndim') and \
           not hasattr(value, 'size') and not hasattr(value, 'mode') and \
           'ndarray' in str(type(value))


def _is_torch_tensor(value: Any) -> bool:
    """Check if value is a torch tensor."""
    return hasattr(value, 'shape') and hasattr(value, 'dtype') and hasattr(value, 'device') and \
           hasattr(value, 'numel') and 'Tensor' in str(type(value))


def _sanitize_result(result: Any) -> tuple[str, bool]:
    """
    Sanitize function result for logging.

    Returns:
        tuple: (sanitized_string, has_large_output)
    """
    if isinstance(result, dict):
        # Extract key info from result dict
        sanitized_items = []
        has_large_output = False

        for key, value in result.items():
            val_str, is_large = _sanitize_value(value)
            sanitized_items.append(f"{key}={val_str}")
            has_large_output = has_large_output or is_large

        return f"{{{', '.join(sanitized_items)}}}", has_large_output

    return _sanitize_value(result)


# =============================================================================
# BATCH PROCESSING DECORATORS (Avoid log spam in loops)
# =============================================================================

def create_batch_decorator(
    module_name: str,
    enable_log: bool = ENABLE_LOG,
    log_batch_start: bool = True,
    log_batch_summary: bool = True,
    log_item_details: bool = False,  # Default: don't log each item (use DEBUG)
):
    """
    Create decorator for batch processing functions.

    Strategy:
    - INFO level: Log batch start/end and summary (total count, success rate, avg time)
    - DEBUG level: Log each item's input/output (for detailed debugging)

    Args:
        module_name: Tên module
        enable_log: Bật/tắt logging
        log_batch_start: Log when batch starts
        log_batch_summary: Log batch summary when complete
        log_item_details: If True, log each item at INFO level (default=False for performance)

    Example:
        @m1_batch()
        def detect_batch(images):
            results = []
            for img in images:
                result = detect(img)
                results.append(result)
            return results
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not enable_log:
                return func(*args, **kwargs)

            module_logger = logging.getLogger(f"WaterMeterAI.{module_name}")
            func_name = func.__name__

            # Log batch start
            if log_batch_start:
                # Estimate batch size from args
                batch_size = _estimate_batch_size(args, kwargs)
                module_logger.info(f"🚦 {func_name} | Starting batch (size≈{batch_size})")

            start_time = time.time()
            results = []
            errors = []

            try:
                # Execute batch function
                result = func(*args, **kwargs)

                # Calculate stats
                elapsed_ms = (time.time() - start_time) * 1000

                # Log batch summary
                if log_batch_summary:
                    success_count = _count_successful_items(result)
                    total_count = _get_total_count(result)
                    success_rate = (success_count / total_count * 100) if total_count > 0 else 0

                    module_logger.info(
                        f"✓ {func_name} | Batch complete: {success_count}/{total_count} successful "
                        f"({success_rate:.1f}%) | Time: {elapsed_ms:.2f}ms "
                        f"({elapsed_ms/total_count:.2f}ms per item)"
                    )

                return result

            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                module_logger.error(
                    f"✗ {func_name} | Batch failed: {type(e).__name__}: {str(e)} | Time: {elapsed_ms:.2f}ms"
                )
                raise

        return wrapper

    return decorator


def _estimate_batch_size(args: tuple, kwargs: dict) -> int:
    """Estimate batch size from function arguments."""
    # Check for common batch patterns
    for arg in args:
        if isinstance(arg, (list, tuple)):
            return len(arg)
        if _is_numpy_array(arg) and arg.ndim > 1:
            return arg.shape[0]  # First dimension is usually batch size

    for value in kwargs.values():
        if isinstance(value, (list, tuple)):
            return len(value)
        if _is_numpy_array(value) and value.ndim > 1:
            return value.shape[0]

    return "unknown"


def _count_successful_items(result: Any) -> int:
    """Count successful items in batch result."""
    if isinstance(result, dict):
        return sum(1 for v in result.values() if v is not None and not isinstance(v, dict))
    elif isinstance(result, (list, tuple)):
        return sum(1 for item in result if item is not None and not isinstance(item, dict) or
                   (isinstance(item, dict) and item.get('success', True)))
    return 1


def _get_total_count(result: Any) -> int:
    """Get total item count in batch result."""
    if isinstance(result, dict):
        return len(result)
    elif isinstance(result, (list, tuple)):
        return len(result)
    return 1


# =============================================================================
# PIPELINE-WRITER DECORATOR
# =============================================================================

def pipeline_step(step_name: str, enable_log: bool = ENABLE_LOG):
    """
    Decorator cho pipeline step (nhiều module chạy串联).

    Example:
        @pipeline_step("M1→M2→M3")
        def run_pipeline(image):
            m1_result = m1_detect(image)
            m2_result = m2_align(m1_result)
            m3_result = m3_detect(m2_result)
            return m3_result
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not enable_log:
                return func(*args, **kwargs)

            pipeline_logger = logging.getLogger(f"WaterMeterAI.Pipeline")
            pipeline_logger.info(f"{'='*60}")
            pipeline_logger.info(f"🚀 Starting Pipeline: {step_name}")
            pipeline_logger.info(f"{'='*60}")

            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                elapsed_sec = time.time() - start_time
                pipeline_logger.info(f"{'='*60}")
                pipeline_logger.info(f"✅ Pipeline Completed: {step_name} | Total Time: {elapsed_sec:.3f}s")
                pipeline_logger.info(f"{'='*60}")

                return result

            except Exception as e:
                elapsed_sec = time.time() - start_time
                pipeline_logger.info(f"{'='*60}")
                pipeline_logger.error(f"❌ Pipeline Failed: {step_name} | Error: {e} | Time: {elapsed_sec:.3f}s")
                pipeline_logger.info(f"{'='*60}")
                raise

        return wrapper

    return decorator


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║         Water Meter AI - Logging Base Framework               ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  Usage Examples:                                               ║
    ║                                                                ║
    ║  from src.common.logging_base import m1_base, m2_base, ...   ║
    ║                                                                ║
    ║  @m1_base()                                                   ║
    ║  def detect_watermeter(image):                                ║
    ║      # Dev chỉ viết logic, không viết log                     ║
    ║      return yolo_model(image)                                 ║
    ║                                                                ║
    ║  @m2_base()                                                   ║
    ║  def align_orientation(image):                                ║
    ║      angle = model.predict(image)                             ║
    ║      return rotate_image(image, angle)                        ║
    ║                                                                ║
    ║  # Tắt logging cho môi trường production                      ║
    ║  # export WATERMETER_ENABLE_LOG=false                        ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
