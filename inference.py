"""
inference.py
============
LLM agent client for the Dockerized CloudOpsEnv server.

Runs a single episode against the HTTP server and prints structured logs.

Environment variables
---------------------
API_BASE_URL   – OpenAI-compatible base URL  (default: "https://api.openai.com/v1")
MODEL_NAME     – Model identifier            (default: "gpt-4.1-mini")
HF_TOKEN       – API key (mandatory)
ENV_BASE_URL   – CloudOpsEnv server URL      (default: "http://localhost:8000")
TASK_TYPE      – Optional task type to force (default: random)

Quick start
-----------
    # Start the server (Docker)
    docker build -t cloudops-env . && docker run -p 8000:8000 cloudops-env

    # Run the agent (separate terminal)
    export HF_TOKEN=sk-...
    python inference.py

Log format
----------
[START] task=<task_name> env=CloudOpsEnv model=<model_name>
[STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<null|msg>
[END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
TASK_TYPE: str | None = os.environ.get("TASK_TYPE")  # optional override

if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN environment variable is required but not set. "
        "Export your Hugging Face / OpenAI API token before running."
    )

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ---------------------------------------------------------------------------
# Env HTTP helpers
# ---------------------------------------------------------------------------

def _post(endpoint: str, payload: dict | None = None) -> dict:
    """POST to the env server and return the parsed JSON body."""
    url = f"{ENV_BASE_URL}{endpoint}"
    response = requests.post(url, json=payload or {}, timeout=30)
    response.raise_for_status()
    return response.json()


def env_reset(task_type: str | None = None) -> dict:
    config: dict[str, Any] = {}
    if task_type:
        config["task_type"] = task_type
    payload = {"config": config} if config else {}
    data = _post("/reset", payload)
    return data["observation"]


def env_step(action: dict) -> dict:
    # /step accepts the action dict directly (tool + args at top level)
    return _post("/step", action)


def env_close() -> None:
    _post("/close")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert cloud-operations engineer acting as an AI agent inside
    a simulation environment.

    Each turn:
    1. Read the latest observation (task description, tool results, history).
    2. Decide which tool to call next.
    3. Reply with ONLY a valid JSON object – no prose, no markdown fences:
       {"tool": "<tool_name>", "args": {<key>: <value>, ...}}

    Strict rules:
    - Use the EXACT tool name from the available tools list.
    - Use the EXACT argument names from the tool schema. Do NOT rename fields.
    - Do NOT omit required fields.
    - If a tool has no required args, use "args": {}.
    - Use diagnostic tools to gather evidence before submitting.
    - When ready, call the appropriate final submission tool.

    Final submission tool schemas (copy these exactly):

      submit_iam_review
        {"tool":"submit_iam_review","args":{"findings":"...","recommendations":"..."}}
        Do NOT use "review", "issues", or any other key.

      submit_finops_report
        {"tool":"submit_finops_report","args":{"summary":"...","recommendations":"..."}}
        Do NOT use "report", "findings", or any other key.

      recommend_instance
        {"tool":"recommend_instance","args":{"new_instance_type":"...","justification":"..."}}
        Do NOT use "instance_type", "recommendation", or any other key.

    Diagnostic example with array args:
      {"tool":"simulate_policy","args":{"principal":"alice","actions":["s3:GetObject"],"resources":["arn:aws:s3:::example-bucket/*"]}}
    """)


def build_user_prompt(obs: dict) -> str:
    """Convert an observation dict into a concise prompt for the LLM."""
    lines: list[str] = [
        f"=== Task: {obs['task_type']} | ID: {obs['task_id']} ===",
        obs["task_description"],
        "",
        f"Step {obs['step']} / {obs['metadata']['max_steps']} "
        f"({obs['metadata']['remaining_steps']} remaining)",
        "",
        "--- Latest tool result ---",
        obs["latest_result"],
        "",
    ]

    if obs["history"]:
        lines.append("--- History (last 3 steps) ---")
        for entry in obs["history"][-3:]:
            lines.append(
                f"  step={entry['step']} tool={entry['tool']} "
                f"args={json.dumps(entry['args'])} reward={entry['reward']}"
            )
            lines.append(f"    result: {entry['result'][:200]}")
        lines.append("")

    lines.append("--- Available tools ---")
    for tool in obs["tools"]:
        lines.append(f"  {tool['name']}: {tool['description']}")
        schema = tool.get("schema", {})
        required = schema.get("required", [])
        props = schema.get("properties", {})
        lines.append(f"    required: {', '.join(required) if required else 'none'}")
        if props:
            lines.append("    properties:")
            for prop_name, prop_def in props.items():
                ptype = prop_def.get("type", "?")
                if ptype == "array":
                    item_type = prop_def.get("items", {}).get("type", "any")
                    ptype = f"array[{item_type}]"
                pdesc = prop_def.get("description", "")
                desc_str = f": {pdesc}" if pdesc else ""
                lines.append(f"      - {prop_name} ({ptype}){desc_str}")
        else:
            lines.append("    properties: none")
    lines.append("")
    lines.append("Respond with ONLY a JSON action object.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call & action parser
# ---------------------------------------------------------------------------

def call_llm(messages: list[dict]) -> str:
    """Call the LLM and return the raw text response."""
    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def parse_action(llm_output: str) -> dict | None:
    """
    Parse LLM output into an action dict.

    Tries strict JSON first, then a lenient search for the first {...} block.
    Returns None if parsing fails.
    """
    # 1. Direct parse
    try:
        action = json.loads(llm_output)
        if isinstance(action, dict) and "tool" in action:
            if "args" not in action:
                action["args"] = {}
            return action
    except json.JSONDecodeError:
        pass

    # 2. Extract first JSON object from fenced or inline text
    start = llm_output.find("{")
    end = llm_output.rfind("}") + 1
    if start != -1 and end > start:
        try:
            action = json.loads(llm_output[start:end])
            if isinstance(action, dict) and "tool" in action:
                if "args" not in action:
                    action["args"] = {}
                return action
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode() -> None:
    """Run a single episode and print structured logs to stdout."""
    # --- Reset ---
    obs = env_reset(task_type=TASK_TYPE)
    task_name = obs.get("task_type", "unknown")

    print(f"[START] task={task_name} env=CloudOpsEnv model={MODEL_NAME}", flush=True)

    step_rewards: list[float] = []
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    done = False
    step_num = 0

    while not done:
        # Build prompt from current observation
        user_content = build_user_prompt(obs)
        messages.append({"role": "user", "content": user_content})

        # Call LLM
        llm_output = call_llm(messages)

        # Append assistant turn (keeps context for multi-turn)
        messages.append({"role": "assistant", "content": llm_output})

        # Parse action
        action = parse_action(llm_output)
        if action is None:
            # Fallback: send a no-op error action so the env records the step
            action = {"tool": "__parse_error__", "args": {"raw": llm_output[:200]}}

        action_str = json.dumps(action)

        # Step environment
        result = env_step(action)
        reward: float = result["reward"]
        done: bool = result["done"]
        error: str | None = result["error"]
        obs = result["observation"]

        step_num += 1
        step_rewards.append(reward)

        # [STEP] log line
        error_str = "null" if error is None else error
        done_str = "true" if done else "false"
        print(
            f"[STEP] step={step_num} action={action_str} "
            f"reward={reward:.2f} done={done_str} error={error_str}",
            flush=True,
        )

    # --- Close ---
    env_close()

    # --- [END] log line ---
    total_reward = sum(step_rewards)
    success = total_reward >= 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)
    success_str = "true" if success else "false"
    print(
        f"[END] success={success_str} steps={step_num} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        run_episode()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Episode aborted by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\n[ERROR] Unhandled exception: {exc}", file=sys.stderr)
        raise
