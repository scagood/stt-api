"""FastAPI application factory and resource lifespan."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import aligner
from .batchworker import build_worker
from .config import (
    AUDIO_WORKERS,
    MODEL_IDLE_TIMEOUT_SEC,
    PRELOAD_MODELS,
    WARMUP,
    WARMUP_SEC,
    WARMUP_TIMEOUT_SEC,
    logger,
)
from .model import evict_idle, get_model, load_model, trim_heap, variant_key, warmup_waveform
from .routes import router

# How often idle models are looked for: one goes at most this long after its timeout.
_IDLE_CHECK_SEC = 60.0


def _shutdown_pool(pool: ThreadPoolExecutor) -> None:
    pool.shutdown(wait=True, cancel_futures=True)


def _exit_without_join(message: str) -> None:
    """Terminate the process while a native call is still running.

    A wedged ORT call cannot be interrupted from Python. Every orderly path
    out — the lifespan's ``worker.stop()`` and the interpreter's own exit —
    joins the executor thread and would hang on it, so leave without one.
    """
    logger.critical(message)
    logging.shutdown()
    os._exit(1)


async def _warmup(app: FastAPI, model_key: str) -> None:
    """Push one synthetic chunk through a preloaded model's real inference path.

    Failure is fatal. The chunk is what every real request looks like, so a
    model that cannot run it would 500 every request while passing readiness.
    A timeout is fatal too, and exits the process outright: cancelling the
    wait leaves the ORT call running on a thread nothing can interrupt, so
    the replica could neither serve past it nor shut down cleanly. The
    orchestrator restarts it. ``PARAKEET_WARMUP_TIMEOUT_SEC`` bounds the wait.
    """
    started = time.perf_counter()
    try:
        await asyncio.wait_for(
            app.state.worker.submit(warmup_waveform(), model_key),
            timeout=WARMUP_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        message = (
            f"warm-up did not finish within {WARMUP_TIMEOUT_SEC:.0f}s; "
            "raise PARAKEET_WARMUP_TIMEOUT_SEC or set PARAKEET_WARMUP=false"
        )
        _exit_without_join(message)
        raise RuntimeError(message) from None  # only reached when _exit_without_join is stubbed
    except Exception as exc:
        raise RuntimeError("warm-up inference failed") from exc
    logger.info("Warm-up of %s completed in %.2fs", model_key, time.perf_counter() - started)


def _unload_idle(timeout: float) -> None:
    """Unload every model and aligner unused for `timeout` seconds, and give
    the memory they held back to the OS."""
    if evict_idle(timeout) + aligner.evict_idle(timeout):
        trim_heap()


async def _unload_idle_periodically(timeout: float) -> None:
    """_unload_idle every _IDLE_CHECK_SEC (or `timeout`, if shorter) until
    cancelled. In a thread: a model load holds the lock eviction takes, through
    its download."""
    while True:
        await asyncio.sleep(min(timeout, _IDLE_CHECK_SEC))
        try:
            await asyncio.to_thread(_unload_idle, timeout)
        except Exception:
            logger.exception("Unloading idle models failed; trying again in %.0fs", _IDLE_CHECK_SEC)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.worker = None
    app.state.audio_pool = ThreadPoolExecutor(
        max_workers=AUDIO_WORKERS, thread_name_prefix="audio"
    )
    # ponytail: one aligner job at a time. On the CPU, on ALIGN_THREADS threads,
    # it can't starve decoding or a CPU Parakeet. On the GPU (fp16 and fp32 on a
    # GPU host) it competes with Parakeet for compute and memory, and only this
    # one-at-a-time bounds it. Word requests, and spoken-number requests with a
    # number to hear (routes._needs_aligner), queue here; add a worker-count knob
    # if their throughput matters.
    app.state.align_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="align")
    evictor = None
    try:
        # An unknown model or quantization fails startup, as it would 400 a request.
        preload = [variant_key(*entry.split(":", 1)) for entry in PRELOAD_MODELS]
        for key in preload:
            logger.info("Lifespan startup: preloading %s", key)
            await asyncio.to_thread(load_model, key)
        app.state.worker = build_worker(get_model)
        await app.state.worker.start()
        if WARMUP and WARMUP_SEC > 0:
            for key in preload:
                await _warmup(app, key)
        app.state.ready = True
        logger.info("Service ready")
        if MODEL_IDLE_TIMEOUT_SEC > 0:
            evictor = asyncio.create_task(
                _unload_idle_periodically(MODEL_IDLE_TIMEOUT_SEC), name="unload_idle"
            )
        yield
    finally:
        app.state.ready = False
        logger.info("Lifespan shutdown")
        if evictor is not None:
            evictor.cancel()
            try:
                await evictor
            except asyncio.CancelledError:
                pass
        if app.state.worker is not None:
            await app.state.worker.stop()
        await asyncio.to_thread(_shutdown_pool, app.state.audio_pool)
        await asyncio.to_thread(_shutdown_pool, app.state.align_pool)


def create_app() -> FastAPI:
    app = FastAPI(
        title="stt-api",
        version="2.0.0",  # x-release-please-version
        description="OpenAI-compatible speech-to-text server on ONNX Runtime.",
        openapi_tags=[
            {"name": "transcription"},
            {"name": "models", "description": "The models and aligners a request may name."},
            {"name": "health"},
        ],
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
