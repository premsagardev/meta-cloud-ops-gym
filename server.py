"""
server.py
=========
FastAPI HTTP server — OpenEnv-compatible API for CloudOpsEnv.

Endpoints
---------
POST /reset   { "config": {"task_type": "..."} }  →  { "observation": {...} }
POST /step    { "tool": "...", "args": {...} }      →  { "observation": {...}, "reward": float, "done": bool, "error": str|null }
POST /close   {}                                   →  { "status": "closed" }
GET  /health                                       →  { "status": "ok" }
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.cloudops_env import CloudOpsEnv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("cloudops.server")

# ---------------------------------------------------------------------------
# App + global state
# ---------------------------------------------------------------------------
app = FastAPI(title="CloudOpsEnv", version="1.0.0")

env_instance: Optional[CloudOpsEnv] = None


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    config: Optional[Dict[str, Any]] = None


class Action(BaseModel):
    tool: str
    args: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset")
def reset_handler(request: ResetRequest = ResetRequest()):
    global env_instance
    try:
        env_instance = CloudOpsEnv()
        obs = env_instance.reset(config=request.config or {})
        log.info("reset task_type=%s task_id=%s", obs.get("task_type"), obs.get("task_id"))
        return {"observation": obs}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("reset error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step")
def step_handler(action: Action):
    if env_instance is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    try:
        result = env_instance.step(action.model_dump())
        log.info(
            "step tool=%s reward=%.4f done=%s error=%s",
            action.tool, result["reward"], result["done"], result["error"],
        )
        return {
            "observation": result["observation"],
            "reward": float(result["reward"]),
            "done": bool(result["done"]),
            "error": result["error"] or None,
        }
    except Exception as exc:
        log.exception("step error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/close")
def close_handler():
    global env_instance
    try:
        if env_instance is not None:
            env_instance.close()
            env_instance = None
        log.info("closed")
        return {"status": "closed"}
    except Exception as exc:
        log.exception("close error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state")
@app.post("/state")
def state_handler():
    """Return current episode metadata without advancing the episode."""
    if env_instance is None:
        return {"task_type": None, "task_id": None, "step_count": 0,
                "max_steps": 0, "done": False, "cumulative_reward": 0.0}
    return env_instance.state()


@app.get("/")
def root_handler():
    return {"status": "ok", "service": "CloudOpsEnv"}


@app.get("/health")
def health_handler():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
