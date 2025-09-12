"""MLIP Logger with Rich formatting."""

import inspect
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from rich.console import Console
from rich.text import Text


class MLIPLogger:
    """Logger class using Rich for formatted output with timing and file origin tracking."""
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize MLIP Logger.
        
        Args:
            console: Rich Console instance. If None, creates a new one.
        """
        self.console = console or Console()
        self.start_time = time.time()
        
    def _get_caller_info(self) -> str:
        """Get the file origin of the calling function."""
        # Go up the stack to find the caller outside this logger
        frame = inspect.currentframe()
        try:
            # Skip current frame and _log_message frame
            caller_frame = frame.f_back.f_back
            if caller_frame:
                file_path = Path(caller_frame.f_code.co_filename)
                # Get relative path from mlip_struct_gen root if possible
                try:
                    parts = file_path.parts
                    if 'mlip_struct_gen' in parts:
                        mlip_index = parts.index('mlip_struct_gen')
                        relative_path = Path(*parts[mlip_index:])
                        return str(relative_path)
                except (ValueError, IndexError):
                    pass
                return file_path.name
            return "unknown"
        finally:
            del frame
    
    def _format_duration(self, elapsed: float) -> str:
        """Format elapsed time as a human-readable duration."""
        if elapsed < 1:
            return f"{elapsed*1000:.0f}ms"
        elif elapsed < 60:
            return f"{elapsed:.1f}s"
        elif elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            return f"{minutes}m{seconds:.1f}s"
        else:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = elapsed % 60
            return f"{hours}h{minutes}m{seconds:.1f}s"
    
    def _log_message(self, level: str, message: str, level_color: str = "white") -> None:
        """
        Internal method to format and print log messages.
        
        Args:
            level: Log level (INFO, DEBUG, WARNING, ERROR)
            message: Message to log
            level_color: Color for the log level
        """
        now = datetime.now()
        elapsed = time.time() - self.start_time
        caller_info = self._get_caller_info()
        
        # Format: [Time] (duration) {file_origin} LEVEL: message
        timestamp = now.strftime("%H:%M:%S")
        duration = self._format_duration(elapsed)
        
        # Create formatted text
        text = Text()
        text.append(f"[{timestamp}] ", style="dim blue")
        text.append(f"({duration}) ", style="dim green")
        text.append(f"{{{caller_info}}} ", style="dim yellow")
        text.append(f"{level}: ", style=f"bold {level_color}")
        text.append(message, style="white")
        
        self.console.print(text)
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self._log_message("INFO", message, "blue")
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self._log_message("DEBUG", message, "magenta")
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self._log_message("WARNING", message, "yellow")
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self._log_message("ERROR", message, "red")
    
    def success(self, message: str) -> None:
        """Log a success message."""
        self._log_message("SUCCESS", message, "green")
    
    def step(self, message: str) -> None:
        """Log a step/process message."""
        self._log_message("STEP", message, "cyan")
    
    def reset_timer(self) -> None:
        """Reset the start time for duration calculations."""
        self.start_time = time.time()
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since logger creation or last reset."""
        return time.time() - self.start_time


# Global logger instance
_global_logger: Optional[MLIPLogger] = None


def get_logger() -> MLIPLogger:
    """
    Get the global logger instance.
    
    Returns:
        Global MLIPLogger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = MLIPLogger()
    return _global_logger


def set_logger(logger: MLIPLogger) -> None:
    """
    Set the global logger instance.
    
    Args:
        logger: MLIPLogger instance to use globally
    """
    global _global_logger
    _global_logger = logger


# Convenience functions for global logger
def info(message: str) -> None:
    """Log an info message using global logger."""
    get_logger().info(message)


def debug(message: str) -> None:
    """Log a debug message using global logger."""
    get_logger().debug(message)


def warning(message: str) -> None:
    """Log a warning message using global logger."""
    get_logger().warning(message)


def error(message: str) -> None:
    """Log an error message using global logger."""
    get_logger().error(message)


def success(message: str) -> None:
    """Log a success message using global logger."""
    get_logger().success(message)


def step(message: str) -> None:
    """Log a step message using global logger."""
    get_logger().step(message)