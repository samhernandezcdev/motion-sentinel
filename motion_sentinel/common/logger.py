"""
Configura structlog para toda la aplicación.

Uso:
    from motion_sentinel.common.logger import get_logger
    log = get_logger(__name__)
    log.info("mensaje", key=value)
"""
import logging
import sys
from typing import Any

import structlog


def setup_logging(level: str = "INFO") -> None:
    """Inicializa structlog con salida legible en consola."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configurar el logging estándar de Python (backend de structlog)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Devuelve un logger con el nombre del módulo como contexto."""
    return structlog.get_logger(name)
