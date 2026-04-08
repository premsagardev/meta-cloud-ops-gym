---
title: CloudOpsEnv
emoji: ☁️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# CloudOpsEnv — Dockerized Cloud Engineering RL Environment

An **OpenEnv-compatible** benchmark environment that simulates three cloud-engineering tasks for LLM agents.

| Task type | Submission tool | Graded on |
|---|---|---|
| `iam_security_review` | `submit_iam_review` | Issues found + recommendations |
| `finops_cost_review` | `submit_finops_report` | Anomalies + actionable recs |
| `ec2_right_sizing` | `recommend_instance` | Correct type + metric reasoning |

---

## Single-command deploy

```bash
docker build -t cloudops-env .
docker run -p 8000:8000 cloudops-env
```

---

## Local run (no Docker)

```bash
pip install -r requirements.txt
python server.py          # starts on :8000
```

---

## Test the endpoints

```bash
# Health check
curl http://localhost:8000/health

# Reset (pick a task type)
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"config": {"task_type": "iam_security_review"}}' | jq .

# Reset with empty body (also valid)
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{}' | jq .

# Step — action sent directly at top level (tool + args)
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"tool": "get_iam_summary", "args": {}}' | jq .

# State — current episode metadata
curl http://localhost:8000/state | jq .

# Close
curl -X POST http://localhost:8000/close | jq .
```

---

## Run the LLM agent

```bash
# Required env vars
export HF_TOKEN=sk-...              # API key (mandatory)
export API_BASE_URL=https://api.openai.com/v1   # OpenAI-compatible base URL
export MODEL_NAME=gpt-4.1-mini      # any OpenAI-compatible model

# Optional
export TASK_TYPE=iam_security_review  # random if omitted
export ENV_BASE_URL=http://localhost:8000

python inference.py
```

---

## Hugging Face Space

This repo is configured as a **Docker Space** (`sdk: docker`). Push to your Space repo and it will build automatically:

```bash
git remote add space https://huggingface.co/spaces/<your-username>/cloudops-env
git push space main
```

Set `HF_TOKEN`, `API_BASE_URL`, and `MODEL_NAME` as Space secrets in the Space settings before running inference.

---

## Validate (OpenEnv)

```bash
pip install openenv-core
openenv validate .
```

---

## Log format

```
[START] task=iam_security_review env=CloudOpsEnv model=gpt-4.1-mini
[STEP] step=1 action={"tool":"get_iam_summary","args":{}} reward=-0.02 done=false error=null
[STEP] step=2 action={"tool":"list_policies","args":{"principal":"alice"}} reward=-0.01 done=false error=null
...
[END] success=true steps=5 rewards=-0.02,-0.01,-0.02,-0.01,0.72
```

`success=true` when cumulative reward ≥ 0.5.

---

## Architecture

```
cloudops-env/
├── openenv.yaml            # OpenEnv manifest (validator)
├── openenv.json            # JSON manifest (compatibility)
├── Dockerfile              # Single-container build
├── server.py               # FastAPI server (POST /reset /step /close, GET /state /health)
├── inference.py            # LLM agent runner (HTTP client)
├── requirements.txt        # Pinned dependencies
└── env/
    ├── __init__.py
    └── cloudops_env.py     # Core env: scenarios, tools, graders, typed models
```

## API reference

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/reset` | POST | `{}` or `{"config": {"task_type": "..."}}` | `{"observation": {...}}` |
| `/step` | POST | `{"tool": "...", "args": {...}}` | `{"observation": {...}, "reward": float, "done": bool, "error": str\|null}` |
| `/state` | GET | — | `{"task_type": ..., "step_count": ..., "done": ..., ...}` |
| `/close` | POST | — | `{"status": "closed"}` |
| `/health` | GET | — | `{"status": "ok"}` |

## Features

- ✅ 3 tasks: IAM security review, FinOps cost review, EC2 right-sizing
- ✅ Schema-driven tools — every tool exposes `required` + `properties` to the agent
- ✅ Typed models: `Action`, `Observation`, `StepResult`, `EpisodeState`
- ✅ `state()` endpoint for episode introspection
- ✅ Backward-compatible arg normalization (`_normalize_final_args`)
- ✅ Dockerized — single `docker run` command
- ✅ Structured OpenEnv API (`/reset`, `/step`, `/state`, `/close`, `/health`)
- ✅ Production logging `[START]`/`[STEP]`/`[END]`
- ✅ All grader scores normalized to `[0.0, 1.0]`
- ✅ Runs on vCPU=2, memory=8GB

## Extending

- **Add scenarios**: append to `IAM_SCENARIOS`, `FINOPS_SCENARIOS`, or `EC2_SCENARIOS` in `env/cloudops_env.py`.
- **Add task types**: add entries to `TOOLS_BY_TASK`, `SCENARIOS_BY_TASK`, implement a grader, wire into `_execute_final`.
- **Multi-session**: replace the global `env_instance` in `server.py` with a `dict[session_id, CloudOpsEnv]`.
