"""
env/server.py
=============
FastAPI HTTP server that exposes CloudOpsEnv over three endpoints:
    POST /reset   – start a new episode
    POST /step    – execute one action
    POST /close   – clean up

A single global env instance is shared across requests (suitable for a
single-user HuggingFace Space; swap for session-keyed dict if needed).
"""

from __future__ import annotations

import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from cloudops_env import CloudOpsEnv  # noqa: E402 – resolved at runtime

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("cloudops.server")

# ---------------------------------------------------------------------------
# Global env instance
# ---------------------------------------------------------------------------
env = CloudOpsEnv()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CloudOpsEnv Server",
    description="OpenEnv-compatible HTTP wrapper for CloudOpsEnv.",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    config: dict[str, Any] | None = Field(
        default=None,
        description="Optional config dict. Supports 'task_type' key.",
    )


class StepRequest(BaseModel):
    action: dict[str, Any] = Field(
        description="Action dict with 'tool' (str) and 'args' (dict) keys.",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset")
def reset(request: ResetRequest) -> JSONResponse:
    """
    Reset the environment and return the initial observation.

    Body (JSON, optional):
        { "config": { "task_type": "iam_security_review" } }
    """
    try:
        observation = env.reset(config=request.config)
        log.info("Episode reset. task_type=%s task_id=%s",
                 observation.get("task_type"), observation.get("task_id"))
        return JSONResponse(content={"observation": observation})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        log.exception("Unexpected error during reset")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


@app.post("/step")
def step(request: StepRequest) -> JSONResponse:
    """
    Execute one action and return observation, reward, done, error.

    Body (JSON):
        { "action": { "tool": "get_iam_summary", "args": {} } }
    """
    try:
        result = env.step(request.action)
        obs = result["observation"]
        log.info(
            "Step %d | tool=%s reward=%.4f done=%s error=%s",
            obs.get("step"),
            request.action.get("tool"),
            result["reward"],
            result["done"],
            result["error"],
        )
        return JSONResponse(content=result)
    except Exception as exc:  # pylint: disable=broad-except
        log.exception("Unexpected error during step")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


@app.post("/close")
def close() -> JSONResponse:
    """Close / clean up the environment state."""
    try:
        env.close()
        log.info("Environment closed.")
        return JSONResponse(content={"status": "closed"})
    except Exception as exc:  # pylint: disable=broad-except
        log.exception("Unexpected error during close")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


@app.get("/health")
def health() -> JSONResponse:
    """Simple health check."""
    return JSONResponse(content={"status": "ok"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
