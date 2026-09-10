"""Health and version endpoints."""

from .api import API_VERSION, bp


@bp.get("/health")
def health():
    """Return a basic server health response."""
    return "OK"


@bp.get("/version")
def version():
    """Return the server version."""
    return API_VERSION
