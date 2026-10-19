"""Health and version endpoints."""
from flask import Response

from .api import API_VERSION, bp


@bp.get("/health")
def health():
    """Return a basic server health response."""
    return Response("OK", status=200, mimetype="text/plain")


@bp.get("/version")
def version():
    """Return the server version."""
    return API_VERSION
