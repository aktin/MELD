"""Health and version endpoints."""
from flask import Response

from InternalDataLoader.db import check_connection
from ModelEnvironment.docker_runtime import check_daemon_access

from .api import API_VERSION, bp


@bp.get("/health")
def health():
    """Check access to the Docker daemon and database."""
    checks = {}
    for name, check in (("docker", check_daemon_access), ("database", check_connection)):
        try:
            checks[name] = "ok" if check() else "unavailable"
        except Exception:
            checks[name] = "unavailable"

    if any(status != "ok" for status in checks.values()):
        return {"status": "unhealthy", "checks": checks}, 503

    return Response("OK", status=200, mimetype="text/plain")


@bp.get("/version")
def version():
    """Return the server version."""
    return API_VERSION
