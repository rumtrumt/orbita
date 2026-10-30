"""Точка входа: ``python -m orbita`` или команда ``orbita``."""

from __future__ import annotations

import uvicorn

from orbita.api.app import create_app
from orbita.config import get_settings
from orbita.container import build_container
from orbita.logging_setup import configure_logging


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = create_app(build_container(settings))
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
