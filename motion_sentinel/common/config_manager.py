"""
Carga la configuración desde un archivo YAML.

Uso:
    from motion_sentinel.common.config_manager import ConfigManager
    cfg = ConfigManager("config/default.yaml")
    source = cfg.get("capture.source", default=0)
"""
from pathlib import Path
from typing import Any

import yaml


class ConfigManager:
    """Lee un archivo YAML y expone sus valores con acceso por ruta de puntos."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: dict[str, Any] = self._load()


    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            raise FileNotFoundError(f"Config file not found: {self._path}")
        with self._path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data or {}


    def reload(self) -> None:
        self._data = self._load()


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """
        Accede a un valor usando notación de puntos.

        Ejemplo: cfg.get("capture.fps", default=30)
        """
        parts = key.split(".")
        node: Any = self._data
        for part in parts:
            if not isinstance(node, dict):
                return default

            if part not in node:
                return default

            node = node[part]

        return node


    @property
    def data(self) -> dict[str, Any]:
        """Devuelve el diccionario completo (solo lectura semántica)."""
        return self._data
