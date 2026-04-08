"""
env/cloudops_env.py
===================
CloudOpsEnv – simulates three cloud-engineering task types:
  • iam_security_review
  • finops_cost_review
  • ec2_right_sizing

Each episode: reset() → N×step(action) → done=True on submission.
All data is hardcoded; no real AWS / billing calls are made.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Typed models (OpenEnv compliance)
# ---------------------------------------------------------------------------

class Action(BaseModel):
    tool: str
    args: dict[str, Any] = {}


class Observation(BaseModel):
    task_type: str
    task_id: str
    task_description: str
    step: int
    tools: list[dict[str, Any]]
    history: list[dict[str, Any]]
    latest_result: str
    metadata: dict[str, Any]


class StepResult(BaseModel):
    observation: dict[str, Any]
    reward: float
    done: bool
    error: Optional[str] = None


class EpisodeState(BaseModel):
    task_type: Optional[str] = None
    task_id: Optional[str] = None
    step_count: int = 0
    max_steps: int = 0
    done: bool = False
    cumulative_reward: float = 0.0

# ---------------------------------------------------------------------------
# Scenario banks
# ---------------------------------------------------------------------------

IAM_SCENARIOS: list[dict] = [
    {
        "task_id": "iam-001",
        "description": (
            "Audit the IAM configuration for the 'data-platform' AWS account. "
            "Several service accounts and human users share the same role. "
            "Identify security issues and recommend fixes."
        ),
        "state": {
            "account_id": "123456789012",
            "account_alias": "data-platform",
            "summary": {
                "Users": 14,
                "Roles": 8,
                "Policies": 22,
                "MFAEnabled": 6,
                "UsersWithoutMFA": 8,
                "AccessKeysOlderThan90Days": 3,
            },
            "principals": {
                "alice": {
                    "type": "user",
                    "mfa_enabled": False,
                    "policies": ["AdminAccess"],
                    "last_login_days": 2,
                },
                "bob": {
                    "type": "user",
                    "mfa_enabled": True,
                    "policies": ["S3FullAccess", "DynamoDBReadOnly"],
                    "last_login_days": 120,
                },
                "svc-etl": {
                    "type": "service_account",
                    "mfa_enabled": False,
                    "policies": ["AdminAccess"],  # over-privileged
                    "last_login_days": 1,
                },
                "carol": {
                    "type": "user",
                    "mfa_enabled": False,
                    "policies": ["IAMFullAccess", "S3FullAccess"],
                    "last_login_days": 5,
                },
            },
            "policies": {
                "AdminAccess": {
                    "policy_id": "pol-admin",
                    "document": {
                        "Effect": "Allow",
                        "Action": "*",
                        "Resource": "*",
                    },
                    "attached_to": ["alice", "svc-etl"],
                },
                "S3FullAccess": {
                    "policy_id": "pol-s3",
                    "document": {
                        "Effect": "Allow",
                        "Action": "s3:*",
                        "Resource": "*",
                    },
                    "attached_to": ["bob", "carol"],
                },
                "IAMFullAccess": {
                    "policy_id": "pol-iam",
                    "document": {
                        "Effect": "Allow",
                        "Action": "iam:*",
                        "Resource": "*",
                    },
                    "attached_to": ["carol"],
                },
                "DynamoDBReadOnly": {
                    "policy_id": "pol-dynamo-ro",
                    "document": {
                        "Effect": "Allow",
                        "Action": ["dynamodb:GetItem", "dynamodb:Query", "dynamodb:Scan"],
                        "Resource": "*",
                    },
                    "attached_to": ["bob"],
                },
            },
        },
        # Ground truth the grader checks against
        "ground_truth": {
            "issues": [
                "mfa_not_enabled_for_privileged_users",
                "service_account_has_admin_access",
                "stale_user_access_bob",
                "carol_has_iam_full_access",
                "wildcard_resource_in_policies",
                "access_keys_older_than_90_days",
            ],
            "ideal_recommendations": [
                "enable_mfa",
                "least_privilege_for_svc_etl",
                "revoke_or_review_bob_access",
                "restrict_carol_iam_permissions",
                "scope_resource_arns",
                "rotate_old_access_keys",
            ],
        },
    },
    {
        "task_id": "iam-002",
        "description": (
            "Review IAM settings for the 'prod-api' account. A recent penetration test "
            "flagged cross-account trust issues. Identify problems and recommend fixes."
        ),
        "state": {
            "account_id": "987654321098",
            "account_alias": "prod-api",
            "summary": {
                "Users": 5,
                "Roles": 12,
                "Policies": 18,
                "MFAEnabled": 5,
                "UsersWithoutMFA": 0,
                "AccessKeysOlderThan90Days": 0,
            },
            "principals": {
                "deploy-role": {
                    "type": "role",
                    "trust_policy": {"Principal": {"AWS": "*"}},  # too broad
                    "policies": ["EC2FullAccess", "S3FullAccess"],
                    "last_login_days": 0,
                },
                "readonly-role": {
                    "type": "role",
                    "trust_policy": {"Principal": {"Service": "ec2.amazonaws.com"}},
                    "policies": ["ReadOnlyAccess"],
                    "last_login_days": 1,
                },
            },
            "policies": {
                "EC2FullAccess": {
                    "policy_id": "pol-ec2",
                    "document": {"Effect": "Allow", "Action": "ec2:*", "Resource": "*"},
                    "attached_to": ["deploy-role"],
                },
                "S3FullAccess": {
                    "policy_id": "pol-s3",
                    "document": {"Effect": "Allow", "Action": "s3:*", "Resource": "*"},
                    "attached_to": ["deploy-role"],
                },
                "ReadOnlyAccess": {
                    "policy_id": "pol-ro",
                    "document": {
                        "Effect": "Allow",
                        "Action": ["ec2:Describe*", "s3:Get*", "s3:List*"],
                        "Resource": "*",
                    },
                    "attached_to": ["readonly-role"],
                },
            },
        },
        "ground_truth": {
            "issues": [
                "deploy_role_trust_policy_allows_any_principal",
                "ec2_full_access_too_broad_for_deploy_role",
                "wildcard_resource_in_policies",
            ],
            "ideal_recommendations": [
                "restrict_trust_policy_to_specific_accounts",
                "scope_deploy_role_to_minimum_required_ec2_actions",
                "scope_resource_arns",
            ],
        },
    },
]

FINOPS_SCENARIOS: list[dict] = [
    {
        "task_id": "finops-001",
        "description": (
            "Analyse the AWS cost breakdown for the 'analytics' team for the last quarter. "
            "Identify anomalies, unused resources, and produce a cost-optimisation report."
        ),
        "state": {
            "team": "analytics",
            "period": "Q1-2024",
            "total_cost_usd": 48320.00,
            "cost_breakdown": {
                "EC2": 22000.00,
                "S3": 3200.00,
                "RDS": 8500.00,
                "Glue": 9400.00,
                "DataTransfer": 5220.00,
            },
            "timeseries": {
                "EC2": [6000, 6500, 9500],    # spike in month 3
                "S3": [1050, 1100, 1050],
                "RDS": [2800, 2800, 2900],
                "Glue": [3100, 3100, 3200],
                "DataTransfer": [1700, 1720, 1800],
            },
            "savings_opportunities": [
                {
                    "type": "Reserved Instances",
                    "service": "EC2",
                    "estimated_savings_usd": 6600,
                    "details": "3 on-demand m5.2xlarge running 24/7 – RI would save ~30 %",
                },
                {
                    "type": "Idle RDS",
                    "service": "RDS",
                    "estimated_savings_usd": 1700,
                    "details": "dev-replica instance has <2 % CPU utilisation over 90 days",
                },
                {
                    "type": "S3 Lifecycle Policy",
                    "service": "S3",
                    "estimated_savings_usd": 640,
                    "details": "raw-logs bucket has 4 TB data with no lifecycle rule",
                },
            ],
            "anomalies": [
                {
                    "service": "EC2",
                    "month": "March",
                    "description": "EC2 cost jumped from $6,500 to $9,500 (+46 %) due to untagged spot instances.",
                }
            ],
        },
        "ground_truth": {
            "anomalies": ["ec2_cost_spike_march", "untagged_spot_instances"],
            "recommendations": [
                "purchase_reserved_instances_for_ec2",
                "terminate_idle_rds_replica",
                "add_s3_lifecycle_policy",
                "enforce_tagging_policy",
            ],
        },
    },
    {
        "task_id": "finops-002",
        "description": (
            "The 'ml-platform' team's AWS bill doubled YoY. Review costs and produce "
            "an actionable FinOps report highlighting the main drivers."
        ),
        "state": {
            "team": "ml-platform",
            "period": "H1-2024",
            "total_cost_usd": 120000.00,
            "cost_breakdown": {
                "SageMaker": 65000.00,
                "EC2": 28000.00,
                "S3": 12000.00,
                "ECR": 6000.00,
                "DataTransfer": 9000.00,
            },
            "timeseries": {
                "SageMaker": [9000, 9500, 10000, 11000, 12500, 13000],
                "EC2": [4000, 4200, 4500, 4800, 5200, 5300],
                "S3": [1800, 1900, 2000, 2100, 2100, 2100],
                "ECR": [900, 950, 1000, 1050, 1050, 1050],
                "DataTransfer": [1300, 1400, 1500, 1600, 1700, 1500],
            },
            "savings_opportunities": [
                {
                    "type": "SageMaker Savings Plans",
                    "service": "SageMaker",
                    "estimated_savings_usd": 19500,
                    "details": "Savings Plans could reduce SageMaker spend by 30 %",
                },
                {
                    "type": "Spot Training Jobs",
                    "service": "SageMaker",
                    "estimated_savings_usd": 8000,
                    "details": "Training jobs are all on-demand; Spot could save ~60 % on training",
                },
                {
                    "type": "ECR Image Cleanup",
                    "service": "ECR",
                    "estimated_savings_usd": 1200,
                    "details": "600 GB of untagged/stale images in ECR",
                },
            ],
            "anomalies": [
                {
                    "service": "SageMaker",
                    "month": "June",
                    "description": "SageMaker endpoints left running over a holiday weekend cost $3,200.",
                }
            ],
        },
        "ground_truth": {
            "anomalies": ["sagemaker_endpoints_left_running", "sagemaker_cost_growth"],
            "recommendations": [
                "adopt_sagemaker_savings_plans",
                "use_spot_for_training_jobs",
                "cleanup_ecr_images",
                "auto_shutdown_idle_endpoints",
            ],
        },
    },
]

EC2_SCENARIOS: list[dict] = [
    {
        "task_id": "ec2-001",
        "description": (
            "The instance 'i-0abc123def456' (m5.4xlarge) is flagged for right-sizing. "
            "Analyse its utilisation metrics and recommend a more cost-effective instance type."
        ),
        "state": {
            "instance_id": "i-0abc123def456",
            "current_type": "m5.4xlarge",
            "current_monthly_cost_usd": 560.00,
            "asg_name": "web-api-asg",
            "metrics": {
                "cpu_avg_pct": 12.4,
                "cpu_p99_pct": 38.0,
                "mem_avg_pct": 24.0,
                "mem_p99_pct": 51.0,
                "network_in_mbps": 45,
                "network_out_mbps": 30,
            },
            "asg_config": {
                "asg_name": "web-api-asg",
                "min_size": 2,
                "max_size": 10,
                "desired_capacity": 4,
                "scaling_policy": "cpu_target_60pct",
            },
            "pricing_options": [
                {"instance_type": "m5.xlarge",  "monthly_cost_usd": 140.0, "vcpu": 4,  "ram_gb": 16},
                {"instance_type": "m5.2xlarge", "monthly_cost_usd": 280.0, "vcpu": 8,  "ram_gb": 32},
                {"instance_type": "m5.4xlarge", "monthly_cost_usd": 560.0, "vcpu": 16, "ram_gb": 64},
                {"instance_type": "m5.8xlarge", "monthly_cost_usd": 1120.0,"vcpu": 32, "ram_gb": 128},
                {"instance_type": "t3.2xlarge", "monthly_cost_usd": 240.0, "vcpu": 8,  "ram_gb": 32},
                {"instance_type": "c5.2xlarge", "monthly_cost_usd": 244.0, "vcpu": 8,  "ram_gb": 16},
            ],
        },
        "ground_truth": {
            "target_instance_type": "m5.2xlarge",
            "acceptable_types": ["m5.2xlarge", "t3.2xlarge"],
            "key_metrics_to_mention": ["cpu_avg_pct", "mem_avg_pct", "cost_savings"],
        },
    },
    {
        "task_id": "ec2-002",
        "description": (
            "Instance 'i-0xyz789ghi012' (r5.4xlarge) is suspected to be memory-oversized. "
            "Evaluate usage and suggest the right instance type."
        ),
        "state": {
            "instance_id": "i-0xyz789ghi012",
            "current_type": "r5.4xlarge",
            "current_monthly_cost_usd": 768.00,
            "asg_name": None,
            "metrics": {
                "cpu_avg_pct": 20.0,
                "cpu_p99_pct": 55.0,
                "mem_avg_pct": 35.0,
                "mem_p99_pct": 62.0,
                "network_in_mbps": 10,
                "network_out_mbps": 8,
            },
            "asg_config": None,
            "pricing_options": [
                {"instance_type": "r5.xlarge",  "monthly_cost_usd": 192.0, "vcpu": 4,  "ram_gb": 32},
                {"instance_type": "r5.2xlarge", "monthly_cost_usd": 384.0, "vcpu": 8,  "ram_gb": 64},
                {"instance_type": "r5.4xlarge", "monthly_cost_usd": 768.0, "vcpu": 16, "ram_gb": 128},
                {"instance_type": "m5.4xlarge", "monthly_cost_usd": 560.0, "vcpu": 16, "ram_gb": 64},
                {"instance_type": "m5.2xlarge", "monthly_cost_usd": 280.0, "vcpu": 8,  "ram_gb": 32},
            ],
        },
        "ground_truth": {
            "target_instance_type": "r5.2xlarge",
            "acceptable_types": ["r5.2xlarge", "m5.4xlarge"],
            "key_metrics_to_mention": ["mem_avg_pct", "cpu_p99_pct", "cost_savings"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool schemas exposed to the agent
# ---------------------------------------------------------------------------

IAM_TOOLS = [
    {
        "name": "get_iam_summary",
        "description": "Return a high-level summary of the IAM configuration (user/role/policy counts, MFA status).",
        "schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_policies",
        "description": "List policies attached to a given principal (user, role, or service account).",
        "schema": {
            "type": "object",
            "properties": {"principal": {"type": "string", "description": "Principal name"}},
            "required": ["principal"],
        },
    },
    {
        "name": "get_policy_document",
        "description": "Retrieve the full policy document for a named policy.",
        "schema": {
            "type": "object",
            "properties": {"policy_id": {"type": "string", "description": "Policy name or ID"}},
            "required": ["policy_id"],
        },
    },
    {
        "name": "simulate_policy",
        "description": "Simulate whether a principal can perform given actions on given resources.",
        "schema": {
            "type": "object",
            "properties": {
                "principal": {"type": "string", "description": "Principal name (user, role, or service account)."},
                "actions": {"type": "array", "items": {"type": "string"}, "description": "Requested AWS actions to simulate (e.g. ['s3:GetObject'])."},
                "resources": {"type": "array", "items": {"type": "string"}, "description": "Resource ARNs to test against (e.g. ['arn:aws:s3:::my-bucket/*'])."},
            },
            "required": ["principal", "actions", "resources"],
        },
    },
    {
        "name": "submit_iam_review",
        "description": "Submit the final IAM security review. This ends the episode.",
        "schema": {
            "type": "object",
            "properties": {
                "findings": {"type": "string", "description": "Security issues found in the IAM configuration."},
                "recommendations": {"type": "string", "description": "Recommended fixes and improvements."},
            },
            "required": ["findings", "recommendations"],
        },
    },
]

FINOPS_TOOLS = [
    {
        "name": "get_cost_breakdown",
        "description": "Get cost breakdown by service for the current scenario.",
        "schema": {
            "type": "object",
            "properties": {
                "granularity": {
                    "type": "string",
                    "enum": ["total", "monthly"],
                    "description": "Aggregation granularity",
                },
                "group_by": {
                    "type": "string",
                    "enum": ["service", "team"],
                    "description": "Dimension to group by",
                },
            },
            "required": ["granularity", "group_by"],
        },
    },
    {
        "name": "get_usage_timeseries",
        "description": "Get monthly cost time series for a specific service.",
        "schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "AWS service name (e.g. EC2, S3, RDS, SageMaker)."},
            },
            "required": ["service"],
        },
    },
    {
        "name": "get_savings_opportunities",
        "description": "List identified savings opportunities with estimated USD savings.",
        "schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "submit_finops_report",
        "description": "Submit the final FinOps cost review report. This ends the episode.",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Summary of the main cost anomalies and observations."},
                "recommendations": {"type": "string", "description": "Recommended cost optimization actions."},
            },
            "required": ["summary", "recommendations"],
        },
    },
]

EC2_TOOLS = [
    {
        "name": "get_instance_metrics",
        "description": "Get CPU, memory, and network utilisation metrics for an instance.",
        "schema": {
            "type": "object",
            "properties": {
                "instance_id": {"type": "string"},
                "period": {
                    "type": "string",
                    "description": "Lookback period, e.g. '30d', '90d'",
                },
            },
            "required": ["instance_id", "period"],
        },
    },
    {
        "name": "get_pricing_options",
        "description": "Get available instance types and their on-demand monthly pricing.",
        "schema": {
            "type": "object",
            "properties": {"instance_id": {"type": "string"}},
            "required": ["instance_id"],
        },
    },
    {
        "name": "get_asg_config",
        "description": "Get Auto Scaling Group configuration for an instance.",
        "schema": {
            "type": "object",
            "properties": {"asg_name": {"type": "string", "description": "ASG name (optional; omit if unknown)."}},
            "required": [],
        },
    },
    {
        "name": "recommend_instance",
        "description": "Submit the final EC2 right-sizing recommendation. This ends the episode.",
        "schema": {
            "type": "object",
            "properties": {
                "new_instance_type": {"type": "string", "description": "Recommended EC2 instance type."},
                "justification": {"type": "string", "description": "Why this instance type is appropriate based on the observed metrics and pricing."},
            },
            "required": ["new_instance_type", "justification"],
        },
    },
]

TOOLS_BY_TASK = {
    "iam_security_review": IAM_TOOLS,
    "finops_cost_review": FINOPS_TOOLS,
    "ec2_right_sizing": EC2_TOOLS,
}

SCENARIOS_BY_TASK = {
    "iam_security_review": IAM_SCENARIOS,
    "finops_cost_review": FINOPS_SCENARIOS,
    "ec2_right_sizing": EC2_SCENARIOS,
}

# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def _keywords_found(text: str, keywords: list[str]) -> float:
    """Return fraction of keywords found (case-insensitive, partial match)."""
    text_lower = text.lower()
    found = sum(1 for kw in keywords if kw.replace("_", " ") in text_lower or kw in text_lower)
    return found / len(keywords) if keywords else 0.0


def grade_iam_review(findings: str, recommendations: str, ground_truth: dict) -> float:
    """
    Score in [0, 1].
    50 % weight on issues found, 50 % on recommendation quality.
    """
    issue_kw = [i.replace("_", " ") for i in ground_truth["issues"]]
    rec_kw = [r.replace("_", " ") for r in ground_truth["ideal_recommendations"]]

    issue_score = _keywords_found(findings, issue_kw)
    rec_score = _keywords_found(recommendations, rec_kw)
    return round(0.5 * issue_score + 0.5 * rec_score, 4)


def grade_finops_report(summary: str, recommendations: str, ground_truth: dict) -> float:
    """
    Score in [0, 1].
    40 % anomaly coverage, 60 % recommendation coverage.
    """
    anomaly_kw = [a.replace("_", " ") for a in ground_truth["anomalies"]]
    rec_kw = [r.replace("_", " ") for r in ground_truth["recommendations"]]

    combined = summary + " " + recommendations
    anomaly_score = _keywords_found(combined, anomaly_kw)
    rec_score = _keywords_found(combined, rec_kw)
    return round(0.4 * anomaly_score + 0.6 * rec_score, 4)


def grade_ec2_recommendation(
    new_instance_type: str,
    justification: str,
    ground_truth: dict,
) -> float:
    """
    Score in [0, 1].
    60 % for correct instance type, 40 % for mentioning right metrics.
    """
    target = ground_truth["target_instance_type"]
    acceptable = ground_truth["acceptable_types"]
    metric_kw = [m.replace("_", " ") for m in ground_truth["key_metrics_to_mention"]]

    if new_instance_type == target:
        type_score = 1.0
    elif new_instance_type in acceptable:
        type_score = 0.75
    else:
        # partial credit for same family
        target_family = target.split(".")[0]
        if new_instance_type.startswith(target_family):
            type_score = 0.3
        else:
            type_score = 0.0

    metric_score = _keywords_found(justification, metric_kw)
    return round(0.6 * type_score + 0.4 * metric_score, 4)


def _normalize_final_args(tool: str, args: dict) -> dict:
    """Temporary backward-compatibility alias handling for older prompts."""
    args = dict(args)  # shallow copy; do not mutate caller's dict
    if tool == "submit_iam_review":
        legacy = args.get("review", "")
        if not args.get("findings") and legacy:
            args["findings"] = legacy
        if not args.get("recommendations") and legacy:
            args["recommendations"] = legacy
    elif tool == "submit_finops_report":
        legacy = args.get("report", "") or args.get("findings", "")
        if not args.get("summary") and legacy:
            args["summary"] = legacy
        if not args.get("recommendations") and legacy:
            args["recommendations"] = legacy
    elif tool == "recommend_instance":
        if not args.get("new_instance_type") and args.get("instance_type"):
            args["new_instance_type"] = args["instance_type"]
        if not args.get("justification"):
            args["justification"] = args.get("reason", "") or args.get("recommendation", "")
    return args


STEP_PENALTY = -0.02          # applied every step
DIAGNOSTIC_BONUS = 0.01       # small reward for valid diagnostic tool calls
MAX_STEPS = 10

FINAL_TOOLS = {"submit_iam_review", "submit_finops_report", "recommend_instance"}
DIAGNOSTIC_TOOLS = {
    "get_iam_summary", "list_policies", "get_policy_document", "simulate_policy",
    "get_cost_breakdown", "get_usage_timeseries", "get_savings_opportunities",
    "get_instance_metrics", "get_pricing_options", "get_asg_config",
}


class CloudOpsEnv:
    """
    OpenEnv-compatible cloud operations environment.

    Usage:
        env = CloudOpsEnv()
        obs = env.reset(config={"task_type": "iam_security_review"})
        result = env.step({"tool": "get_iam_summary", "args": {}})
        env.close()
    """

    def __init__(self) -> None:
        self._scenario: dict | None = None
        self._task_type: str | None = None
        self._ground_truth: dict | None = None
        self._step_count: int = 0
        self._history: list[dict] = []
        self._cumulative_step_reward: float = 0.0
        self._done: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, config: dict | None = None) -> dict:
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        config : dict, optional
            May contain ``task_type`` key. If absent, a task type is sampled
            uniformly at random.

        Returns
        -------
        dict
            Initial observation.
        """
        config = config or {}
        task_type = config.get("task_type")

        if task_type is None:
            task_type = random.choice(list(SCENARIOS_BY_TASK.keys()))

        if task_type not in SCENARIOS_BY_TASK:
            raise ValueError(
                f"Unknown task_type '{task_type}'. "
                f"Choose from {list(SCENARIOS_BY_TASK.keys())}"
            )

        scenario_pool = SCENARIOS_BY_TASK[task_type]
        scenario = deepcopy(random.choice(scenario_pool))

        self._task_type = task_type
        self._scenario = scenario
        self._ground_truth = scenario.pop("ground_truth")  # hide from agent
        self._step_count = 0
        self._history = []
        self._cumulative_step_reward = 0.0
        self._done = False

        return self._build_observation(
            latest_result="Episode started. Use the available tools to investigate the scenario."
        )

    def step(self, action: dict) -> dict:
        """
        Execute one action.

        Parameters
        ----------
        action : dict
            Must contain ``tool`` (str) and ``args`` (dict).

        Returns
        -------
        dict with keys: observation, reward, done, error
        """
        if self._done:
            return {
                "observation": self._build_observation("Episode already finished."),
                "reward": 0.0,
                "done": True,
                "error": "Episode is already done. Call reset() to start a new episode.",
            }

        tool = action.get("tool", "")
        args = action.get("args", {})
        error: str | None = None
        reward: float = STEP_PENALTY
        done = False

        # --- Execute tool ---
        if tool in DIAGNOSTIC_TOOLS:
            result, exec_error = self._execute_diagnostic(tool, args)
            if exec_error:
                error = exec_error
            else:
                reward += DIAGNOSTIC_BONUS
            latest_result = result
        elif tool in FINAL_TOOLS:
            final_score, exec_error = self._execute_final(tool, args)
            if exec_error:
                error = exec_error
                latest_result = f"Submission error: {exec_error}"
            else:
                reward += final_score
                done = True
                self._done = True
                latest_result = (
                    f"Submission accepted. Final score: {final_score:.4f}. Episode complete."
                )
        else:
            error = f"Unknown tool '{tool}'. Check the tools list."
            latest_result = error

        self._step_count += 1
        self._cumulative_step_reward += reward

        # Append to history
        self._history.append(
            {
                "step": self._step_count,
                "tool": tool,
                "args": args,
                "result": latest_result,
                "reward": round(reward, 4),
                "error": error,
            }
        )

        # Force-done if max steps reached
        if self._step_count >= MAX_STEPS and not done:
            done = True
            self._done = True
            latest_result += " | Max steps reached; episode terminated."

        obs = self._build_observation(latest_result)
        return {
            "observation": obs,
            "reward": round(reward, 4),
            "done": done,
            "error": error,
        }

    def state(self) -> dict:
        """
        Return current episode metadata without advancing the episode.
        Safe to call at any point after reset().
        """
        return EpisodeState(
            task_type=self._task_type,
            task_id=self._scenario.get("task_id") if self._scenario else None,
            step_count=self._step_count,
            max_steps=MAX_STEPS,
            done=self._done,
            cumulative_reward=round(self._cumulative_step_reward, 4),
        ).model_dump()

    def close(self) -> None:
        """Release any resources (no-op in this implementation)."""
        self._scenario = None
        self._ground_truth = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self, latest_result: str) -> dict:
        remaining = MAX_STEPS - self._step_count
        return {
            "task_type": self._task_type,
            "task_id": self._scenario.get("task_id", "unknown"),
            "task_description": self._scenario.get("description", ""),
            "step": self._step_count,
            "tools": TOOLS_BY_TASK.get(self._task_type, []),
            "history": deepcopy(self._history),
            "latest_result": latest_result,
            "metadata": {
                "max_steps": MAX_STEPS,
                "remaining_steps": remaining,
            },
        }

    # ------------------------------------------------------------------
    # Diagnostic tool executors
    # ------------------------------------------------------------------

    def _execute_diagnostic(self, tool: str, args: dict) -> tuple[str, str | None]:
        """
        Dispatch to the correct diagnostic handler.

        Returns (result_string, error_or_None).
        """
        try:
            handler = {
                # IAM
                "get_iam_summary": self._iam_get_summary,
                "list_policies": self._iam_list_policies,
                "get_policy_document": self._iam_get_policy_document,
                "simulate_policy": self._iam_simulate_policy,
                # FinOps
                "get_cost_breakdown": self._finops_get_cost_breakdown,
                "get_usage_timeseries": self._finops_get_usage_timeseries,
                "get_savings_opportunities": self._finops_get_savings_opportunities,
                # EC2
                "get_instance_metrics": self._ec2_get_instance_metrics,
                "get_pricing_options": self._ec2_get_pricing_options,
                "get_asg_config": self._ec2_get_asg_config,
            }.get(tool)

            if handler is None:
                return "", f"No handler for diagnostic tool '{tool}'"
            return handler(args), None
        except Exception as exc:  # pylint: disable=broad-except
            return "", f"Tool execution error: {exc}"

    def _execute_final(self, tool: str, args: dict) -> tuple[float, str | None]:
        """
        Execute a submission/final tool and return (score, error_or_None).
        """
        try:
            gt = self._ground_truth
            args = _normalize_final_args(tool, args)

            if tool == "submit_iam_review":
                findings = args.get("findings", "")
                recs = args.get("recommendations", "")
                if not findings or not recs:
                    return 0.0, "submit_iam_review requires 'findings' and 'recommendations'"
                return grade_iam_review(findings, recs, gt), None

            if tool == "submit_finops_report":
                summary = args.get("summary", "")
                recs = args.get("recommendations", "")
                if not summary or not recs:
                    return 0.0, "submit_finops_report requires 'summary' and 'recommendations'"
                return grade_finops_report(summary, recs, gt), None

            if tool == "recommend_instance":
                new_type = args.get("new_instance_type", "")
                justification = args.get("justification", "")
                if not new_type or not justification:
                    return 0.0, "recommend_instance requires 'new_instance_type' and 'justification'"
                return grade_ec2_recommendation(new_type, justification, gt), None

            return 0.0, f"Unknown final tool '{tool}'"
        except Exception as exc:  # pylint: disable=broad-except
            return 0.0, f"Grader error: {exc}"

    # ------------------------------------------------------------------
    # IAM tool implementations
    # ------------------------------------------------------------------

    def _iam_get_summary(self, _args: dict) -> str:
        state = self._scenario["state"]
        s = state["summary"]
        return (
            f"IAM Summary for account '{state['account_alias']}' ({state['account_id']}):\n"
            + "\n".join(f"  {k}: {v}" for k, v in s.items())
        )

    def _iam_list_policies(self, args: dict) -> str:
        principal = args.get("principal", "")
        state = self._scenario["state"]
        principals = state.get("principals", {})
        if principal not in principals:
            return f"Principal '{principal}' not found. Available: {list(principals.keys())}"
        p = principals[principal]
        policies = p.get("policies", [])
        mfa = p.get("mfa_enabled", False)
        last_login = p.get("last_login_days", "N/A")
        return (
            f"Principal: {principal} (type={p['type']})\n"
            f"  MFA enabled: {mfa}\n"
            f"  Last login: {last_login} days ago\n"
            f"  Attached policies: {policies}"
        )

    def _iam_get_policy_document(self, args: dict) -> str:
        policy_id = args.get("policy_id", "")
        state = self._scenario["state"]
        policies = state.get("policies", {})
        if policy_id not in policies:
            return f"Policy '{policy_id}' not found. Available: {list(policies.keys())}"
        p = policies[policy_id]
        doc = p["document"]
        attached = p.get("attached_to", [])
        return (
            f"Policy: {policy_id}\n"
            f"  Document: {doc}\n"
            f"  Attached to: {attached}"
        )

    def _iam_simulate_policy(self, args: dict) -> str:
        principal = args.get("principal", "")
        actions = args.get("actions", [])
        resources = args.get("resources", [])
        state = self._scenario["state"]
        principals = state.get("principals", {})

        if principal not in principals:
            return f"Principal '{principal}' not found."

        p = principals[principal]
        policies_data = state.get("policies", {})
        all_actions_allowed: list[str] = []
        for pol_name in p.get("policies", []):
            doc = policies_data.get(pol_name, {}).get("document", {})
            pol_action = doc.get("Action", [])
            if pol_action == "*" or pol_action == ["*"]:
                all_actions_allowed = actions
                break
            if isinstance(pol_action, str):
                pol_action = [pol_action]
            for req_action in actions:
                service = req_action.split(":")[0] if ":" in req_action else req_action
                if req_action in pol_action or f"{service}:*" in pol_action:
                    if req_action not in all_actions_allowed:
                        all_actions_allowed.append(req_action)

        denied = [a for a in actions if a not in all_actions_allowed]
        return (
            f"Simulation for {principal} on resources {resources}:\n"
            f"  ALLOWED: {all_actions_allowed}\n"
            f"  DENIED:  {denied}"
        )

    # ------------------------------------------------------------------
    # FinOps tool implementations
    # ------------------------------------------------------------------

    def _finops_get_cost_breakdown(self, args: dict) -> str:
        state = self._scenario["state"]
        granularity = args.get("granularity", "total")
        group_by = args.get("group_by", "service")
        breakdown = state.get("cost_breakdown", {})
        total = state.get("total_cost_usd", 0)

        if granularity == "monthly" and "timeseries" in state:
            ts = state["timeseries"]
            lines = []
            for svc, monthly in ts.items():
                lines.append(f"  {svc}: {monthly}")
            return (
                f"Monthly cost breakdown (group_by={group_by}) "
                f"for {state.get('team','?')} – {state.get('period','?')}:\n"
                + "\n".join(lines)
            )

        lines = [f"  {svc}: ${cost:,.2f}" for svc, cost in breakdown.items()]
        return (
            f"Total cost breakdown (group_by={group_by}) "
            f"for {state.get('team','?')} – {state.get('period','?')}:\n"
            + "\n".join(lines)
            + f"\n  TOTAL: ${total:,.2f}"
        )

    def _finops_get_usage_timeseries(self, args: dict) -> str:
        service = args.get("service", "")
        state = self._scenario["state"]
        ts = state.get("timeseries", {})
        if service not in ts:
            return f"No timeseries data for service '{service}'. Available: {list(ts.keys())}"
        values = ts[service]
        return (
            f"Monthly costs for {service} over {state.get('period','?')}:\n"
            + "  " + " → ".join(f"${v:,}" for v in values)
            + f"\n  Min: ${min(values):,}  Max: ${max(values):,}  "
            + f"Trend: {'↑' if values[-1] > values[0] else '↓' if values[-1] < values[0] else '→'}"
        )

    def _finops_get_savings_opportunities(self, _args: dict) -> str:
        state = self._scenario["state"]
        opps = state.get("savings_opportunities", [])
        if not opps:
            return "No savings opportunities identified."
        lines = []
        for o in opps:
            lines.append(
                f"  [{o['type']}] ({o['service']}) "
                f"~${o['estimated_savings_usd']:,}/mo: {o['details']}"
            )
        anomalies = state.get("anomalies", [])
        anom_lines = []
        for a in anomalies:
            anom_lines.append(f"  [{a['service']} – {a['month']}] {a['description']}")
        result = "Savings Opportunities:\n" + "\n".join(lines)
        if anom_lines:
            result += "\n\nCost Anomalies:\n" + "\n".join(anom_lines)
        return result

    # ------------------------------------------------------------------
    # EC2 tool implementations
    # ------------------------------------------------------------------

    def _ec2_get_instance_metrics(self, args: dict) -> str:
        instance_id = args.get("instance_id", "")
        state = self._scenario["state"]
        if instance_id and instance_id != state.get("instance_id"):
            return f"Instance '{instance_id}' not found in this scenario."
        m = state.get("metrics", {})
        return (
            f"Metrics for {state['instance_id']} (type={state['current_type']}, "
            f"cost=${state['current_monthly_cost_usd']:,.2f}/mo):\n"
            f"  CPU avg: {m['cpu_avg_pct']}%  p99: {m['cpu_p99_pct']}%\n"
            f"  Memory avg: {m['mem_avg_pct']}%  p99: {m['mem_p99_pct']}%\n"
            f"  Network in: {m['network_in_mbps']} Mbps  out: {m['network_out_mbps']} Mbps"
        )

    def _ec2_get_pricing_options(self, args: dict) -> str:
        state = self._scenario["state"]
        options = state.get("pricing_options", [])
        lines = [
            f"  {o['instance_type']:15s} vCPU={o['vcpu']:2d}  RAM={o['ram_gb']:4d}GB  "
            f"${o['monthly_cost_usd']:7.2f}/mo"
            for o in options
        ]
        return (
            f"Pricing options for workload type (current: {state['current_type']}):\n"
            + "\n".join(lines)
        )

    def _ec2_get_asg_config(self, args: dict) -> str:
        asg_name = args.get("asg_name", "")
        state = self._scenario["state"]
        cfg = state.get("asg_config")
        if cfg is None:
            return "This instance is not part of an Auto Scaling Group."
        if asg_name and asg_name != cfg.get("asg_name"):
            return f"ASG '{asg_name}' not found. This instance belongs to '{cfg['asg_name']}'."
        return (
            f"ASG Config for '{cfg['asg_name']}':\n"
            f"  Min: {cfg['min_size']}  Desired: {cfg['desired_capacity']}  Max: {cfg['max_size']}\n"
            f"  Scaling policy: {cfg['scaling_policy']}"
        )
