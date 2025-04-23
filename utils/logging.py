# GenAI is used for rephrasing comments and debugging.
import logging
from pathlib import Path

def setup_logging(log_dir: Path, log_name="experiment.log", level=logging.INFO):
    """
    Initialize root logger with file + console handlers if not already initialized.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name

    # Avoid re-adding handlers if already configured
    if not logging.getLogger().hasHandlers():
        formatter = logging.Formatter('%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logging.basicConfig(level=level, handlers=[file_handler, stream_handler])
        logging.getLogger().info(f"Logging initialized: {log_path}")
