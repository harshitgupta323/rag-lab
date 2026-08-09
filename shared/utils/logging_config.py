import logging
from pathlib import Path


def setup_logging(
    log_dir: Path,
    log_filename: str = "rag_lab.log",
    level: int = logging.INFO,
) -> None:
    """
    Configure application-wide logging.

    Logs are written to:
        1. Console
        2. Log file

    Parameters
    ----------
    log_dir : Path
        Directory where the log file should be stored.

    log_filename : str
        Name of the log file.

    level : int
        Logging level.
    """

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = log_dir / log_filename

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()

    root_logger.setLevel(level)

    # Prevent duplicate handlers if setup_logging()
    # is called more than once.
    if root_logger.handlers:
        return

    # -------------------------------------------------
    # Console Handler
    # -------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # -------------------------------------------------
    # File Handler
    # -------------------------------------------------

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # -------------------------------------------------
    # Register handlers
    # -------------------------------------------------

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for a specific module.
    """

    return logging.getLogger(name)