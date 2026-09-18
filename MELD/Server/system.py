"""Health and version endpoints."""
from flask import Response

from .api import API_VERSION, bp


@bp.get("/health")
def health():
    """Return a basic server health response."""
    return Response(status_code=200)


@bp.get("/version")
def version():
    """Return the server version."""
    return API_VERSION
