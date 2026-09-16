from __future__ import annotations
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("ragdms.errors")

class DocumentNotFoundError(Exception): ...
class UnsupportedFileError(Exception): ...
class ProcessingError(Exception): ...

def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DocumentNotFoundError)
    async def document_not_found_handler(request: Request, exc: DocumentNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc) or "Document not found"})

    @app.exception_handler(UnsupportedFileError)
    async def unsupported_file_handler(request: Request, exc: UnsupportedFileError):
        return JSONResponse(status_code=400, content={"detail": str(exc) or "Unsupported file format"})

    @app.exception_handler(ProcessingError)
    async def processing_error_handler(request: Request, exc: ProcessingError):
        return JSONResponse(status_code=500, content={"detail": str(exc) or "Processing error"})

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
