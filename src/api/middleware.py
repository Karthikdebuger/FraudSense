"""
FraudSense API Middleware

CORS, error handling, and request logging.
"""

import time
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("fraudsense")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests with timing."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        try:
            response = await call_next(request)
            elapsed = time.time() - start
            logger.info(
                f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.3f}s)"
            )
            return response
        except Exception as e:
            elapsed = time.time() - start
            logger.error(
                f"{request.method} {request.url.path} → 500 ({elapsed:.3f}s) Error: {e}"
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "error": str(e)},
            )
