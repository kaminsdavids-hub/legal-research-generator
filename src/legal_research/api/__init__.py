"""FastAPI application exposing the pipeline to the frontend (spec §3)."""

from __future__ import annotations

from .app import app, main

__all__ = ["app", "main"]
