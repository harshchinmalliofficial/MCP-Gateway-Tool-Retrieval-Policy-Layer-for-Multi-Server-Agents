"""Tool sources for the gateway.

Three origins, always clearly labelled:

* ``real``     - pulled live from a public MCP server over stdio (filesystem +
                 "everything" reference servers).  Best-effort: if npx / the
                 network is unavailable we skip them and say so.
* ``curated``  - hand-written but realistic tool schemas for common
                 infra/SaaS actions.  The benchmark's "correct answers" point
                 at these so the test set is stable no matter what the network
                 does.
* ``synthetic``- machine-generated enterprise "distractor" tools (billing, HR,
                 devops, analytics, ...) that simulate real tool sprawl.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import config
from gateway.mcp_client import MCPServerSpec, fetch_tools_from_server

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Tool:
    """A single callable tool exposed to the agent."""

    name: str
    description: str
    parameters: dict[str, Any]
    server: str
    origin: str  # "real" | "curated" | "synthetic"
    keywords: str = ""  # extra synonyms folded into the retrieval text only

    def embedding_text(self) -> str:
        """Text fed to the embedding model - name + description + keywords."""
        extra = f" {self.keywords}" if self.keywords else ""
        return f"{self.name}: {self.description}{extra}"

    def lexical_text(self) -> str:
        """Text used for the lexical half of hybrid retrieval - adds the
        server domain and parameter names as extra keyword surface."""
        params = " ".join((self.parameters or {}).get("properties", {}).keys())
        domain = self.server.split(":")[-1].split(" ")[0]
        return f"{self.name} {self.description} {domain} {params} {self.keywords}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def llm_schema(self) -> dict[str, Any]:
        """Compact JSON schema handed to the LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# --------------------------------------------------------------------------- #
# Real MCP servers (best-effort)
# --------------------------------------------------------------------------- #

REAL_SERVERS = [
    MCPServerSpec(
        key="filesystem",
        command=["npx", "-y", "@modelcontextprotocol/server-filesystem", str(config.ROOT)],
        note="Official reference server: sandboxed file operations.",
    ),
    MCPServerSpec(
        key="everything",
        command=["npx", "-y", "@modelcontextprotocol/server-everything"],
        note="Official reference server: demo/echo/util tools.",
    ),
]

_REAL_CACHE = config.DATA_DIR / "real_tools_cache.json"


def _normalise_real(raw: dict[str, Any], server_key: str) -> Tool:
    return Tool(
        name=raw["name"],
        description=(raw.get("description") or "").strip() or raw["name"],
        parameters=raw.get("inputSchema") or {"type": "object", "properties": {}},
        server=f"{server_key} (real MCP server)",
        origin="real",
    )


def load_real_tools(allow_network: bool = True, timeout: float = 90.0) -> list[Tool]:
    """Try each real server once; fall back to a local cache; else return []."""
    if allow_network:
        collected: list[Tool] = []
        ok_servers: list[str] = []
        for spec in REAL_SERVERS:
            try:
                raw_tools = fetch_tools_from_server(spec, startup_timeout=timeout)
                collected.extend(_normalise_real(t, spec.key) for t in raw_tools)
                ok_servers.append(spec.key)
            except Exception as exc:  # noqa: BLE001 - best effort by design
                print(f"  [real MCP] {spec.key}: unavailable ({type(exc).__name__}: {exc})")
        if collected:
            _REAL_CACHE.write_text(
                json.dumps([t.to_dict() for t in collected], indent=2)
            )
            print(f"  [real MCP] connected: {', '.join(ok_servers)} "
                  f"-> {len(collected)} real tools")
            return collected

    if _REAL_CACHE.exists():
        cached = [Tool(**d) for d in json.loads(_REAL_CACHE.read_text())]
        print(f"  [real MCP] using cached schemas -> {len(cached)} real tools")
        return cached

    print("  [real MCP] no real servers reachable and no cache - continuing "
          "with curated + synthetic tools only")
    return []


# --------------------------------------------------------------------------- #
# Curated realistic tools - the benchmark's ground-truth targets
# --------------------------------------------------------------------------- #

_STR = {"type": "string"}
_INT = {"type": "integer"}
_BOOL = {"type": "boolean"}


def _obj(**props: Any) -> dict[str, Any]:
    required = [k for k, v in props.items() if not v.pop("_opt", False)]
    return {"type": "object", "properties": props, "required": required}


CURATED_TOOLS: list[Tool] = [
    Tool("fs.read_file", "Read and return the full contents of a text file from the local filesystem.",
         _obj(path=dict(_STR)), "curated:filesystem", "curated",
         "view show display open cat print see what is inside contents of a file source code"),
    Tool("fs.write_file", "Create or overwrite a file on the local filesystem with the given text.",
         _obj(path=dict(_STR), content=dict(_STR)), "curated:filesystem", "curated",
         "save put store add write new file into to disk"),
    Tool("fs.list_directory", "List the files and sub-directories inside a directory.",
         _obj(path=dict(_STR)), "curated:filesystem", "curated",
         "ls dir folder contents what files are in show directory listing"),
    Tool("fs.search_files", "Recursively find files whose name matches a glob pattern.",
         _obj(root=dict(_STR), pattern=dict(_STR)), "curated:filesystem", "curated",
         "locate grep find where is every file matching pattern"),
    Tool("git.commit", "Stage all changes and create a git commit with a message on the current branch.",
         _obj(message=dict(_STR)), "curated:git", "curated",
         "check in save changes version control commit message"),
    Tool("git.create_branch", "Create a new git branch from the current HEAD and switch to it.",
         _obj(name=dict(_STR)), "curated:git", "curated",
         "cut a branch checkout new feature branch off main"),
    Tool("git.open_pull_request", "Open a GitHub pull request (PR) from a branch into the default branch for review.",
         _obj(branch=dict(_STR), title=dict(_STR), body={**_STR, "_opt": True}),
         "curated:github", "curated",
         "raise open a PR merge request propose changes for review github"),
    Tool("http.get", "Perform an HTTP GET request to a URL and return the response body.",
         _obj(url=dict(_STR)), "curated:http", "curated",
         "fetch download call hit an endpoint api url webpage status check"),
    Tool("http.post_json", "Send an HTTP POST request with a JSON body to a URL.",
         _obj(url=dict(_STR), body={"type": "object"}), "curated:http", "curated",
         "post send payload json body to endpoint api webhook"),
    Tool("sql.run_query", "Execute a read-only SQL SELECT query against the analytics data warehouse.",
         _obj(query=dict(_STR)), "curated:warehouse", "curated",
         "select count sum group by run a query report numbers rows from database warehouse"),
    Tool("db.backup", "Trigger an on-demand snapshot backup of a production database.",
         _obj(database=dict(_STR)), "curated:database", "curated",
         "back up dump snapshot database before migration disaster recovery"),
    Tool("email.send", "Send a plain-text or HTML email to one or more recipients.",
         _obj(to={"type": "array", "items": _STR}, subject=dict(_STR), body=dict(_STR)),
         "curated:email", "curated",
         "email mail message write to someone notify by email distro list inbox"),
    Tool("calendar.create_event", "Schedule a calendar event / meeting with a title, time window and attendees.",
         _obj(title=dict(_STR), start=dict(_STR), end=dict(_STR),
              attendees={"type": "array", "items": _STR, "_opt": True}),
         "curated:calendar", "curated",
         "schedule book set up a meeting invite appointment sync call on my calendar tomorrow friday"),
    Tool("slack.post_message", "Post a message to a Slack channel or direct message.",
         _obj(channel=dict(_STR), text=dict(_STR)), "curated:slack", "curated",
         "slack chat post drop a note ping the team message channel"),
    Tool("pagerduty.create_incident", "Open a PagerDuty incident and page the on-call engineer.",
         _obj(service=dict(_STR), title=dict(_STR), urgency={**_STR, "_opt": True}),
         "curated:pagerduty", "curated",
         "page alert wake the on call open an incident outage service is down 500s errors urgency"),
    Tool("k8s.rollout_restart", "Restart all pods of a Kubernetes deployment (rolling restart).",
         _obj(namespace=dict(_STR), deployment=dict(_STR)), "curated:kubernetes", "curated",
         "restart bounce cycle pods rolling restart kubernetes k8s deployment"),
    Tool("k8s.scale_deployment", "Set the replica count of a Kubernetes deployment (scale up or down).",
         _obj(namespace=dict(_STR), deployment=dict(_STR), replicas=dict(_INT)),
         "curated:kubernetes", "curated",
         "scale up down bump replicas pods count kubernetes k8s deployment to zero"),
    Tool("aws.restart_ec2_instance", "Reboot a specific AWS EC2 instance by instance id.",
         _obj(instance_id=dict(_STR)), "curated:aws", "curated",
         "reboot restart bounce ec2 vm instance aws machine"),
    Tool("aws.upload_to_s3", "Upload a local file to an S3 bucket at a given key.",
         _obj(bucket=dict(_STR), key=dict(_STR), path=dict(_STR)), "curated:aws", "curated",
         "upload push copy file artifact to s3 bucket object storage aws"),
    Tool("stripe.create_refund", "Issue a refund for a Stripe payment / charge id (give the customer their money back).",
         _obj(charge_id=dict(_STR), amount={**_INT, "_opt": True}), "curated:stripe", "curated",
         "refund reverse pay back money to customer stripe charge payment duplicate"),
    Tool("stripe.list_failed_payments", "List Stripe payments that failed within a date range.",
         _obj(since=dict(_STR), until=dict(_STR)), "curated:stripe", "curated",
         "list show failed declined payments charges stripe last week billing"),
    Tool("jira.create_issue", "Create a new Jira ticket / bug in a project with a summary and description.",
         _obj(project=dict(_STR), summary=dict(_STR), description={**_STR, "_opt": True}),
         "curated:jira", "curated",
         "file open raise a jira ticket bug story task in project"),
    Tool("jira.transition_issue", "Move an existing Jira issue to a new workflow status (e.g. In Progress, Done).",
         _obj(issue_key=dict(_STR), status=dict(_STR)), "curated:jira", "curated",
         "move mark set transition a jira ticket issue to done in progress closed status"),
    Tool("datadog.query_metric", "Query and graph a time series of a Datadog metric over a time window.",
         _obj(metric=dict(_STR), from_=dict(_STR), to=dict(_STR)), "curated:datadog", "curated",
         "graph plot chart query a metric latency p99 cpu datadog observability last hours"),
    Tool("dns.upsert_record", "Create or update a DNS record in a hosted zone (point a hostname at an address).",
         _obj(zone=dict(_STR), name=dict(_STR), type=dict(_STR), value=dict(_STR)),
         "curated:dns", "curated",
         "point add update a dns record a cname hostname domain at ip address load balancer"),
    Tool("feature_flag.toggle", "Enable or disable a feature flag for an environment.",
         _obj(flag=dict(_STR), environment=dict(_STR), enabled=dict(_BOOL)),
         "curated:launchdarkly", "curated",
         "turn on off enable disable a feature flag toggle rollout in production staging"),
    Tool("secrets.rotate", "Rotate a stored secret / credential and return the new version id.",
         _obj(name=dict(_STR)), "curated:vault", "curated",
         "rotate cycle regenerate a secret credential password api key that may be leaked vault"),
]

CURATED_TOOL_NAMES = {t.name for t in CURATED_TOOLS}


# --------------------------------------------------------------------------- #
# Synthetic distractor generator - simulated enterprise tool sprawl
# --------------------------------------------------------------------------- #

_DOMAINS: dict[str, dict[str, list[str]]] = {
    "billing": {
        "objects": ["invoice", "subscription", "credit_note", "tax_rate", "coupon",
                    "payment_method", "dunning_rule", "price_plan"],
        "verbs": ["create", "void", "finalize", "list", "update", "export", "reconcile"],
    },
    "devops": {
        "objects": ["pipeline", "runner", "artifact", "release", "environment",
                    "deploy_lock", "build_cache", "webhook"],
        "verbs": ["trigger", "cancel", "retry", "promote", "list", "purge", "describe"],
    },
    "hr": {
        "objects": ["employee", "onboarding_task", "pto_request", "payroll_run",
                    "org_chart", "benefit_plan", "performance_review"],
        "verbs": ["create", "approve", "reject", "list", "update", "archive", "export"],
    },
    "analytics": {
        "objects": ["dashboard", "funnel", "cohort", "segment", "report_schedule",
                    "event_schema", "attribution_model"],
        "verbs": ["build", "refresh", "list", "clone", "share", "delete", "snapshot"],
    },
    "crm": {
        "objects": ["lead", "opportunity", "account", "contact", "campaign",
                    "quote", "support_ticket"],
        "verbs": ["create", "convert", "assign", "list", "merge", "close", "escalate"],
    },
    "security": {
        "objects": ["access_review", "sso_connection", "audit_export", "device",
                    "vulnerability", "firewall_rule", "api_token"],
        "verbs": ["create", "revoke", "list", "approve", "quarantine", "scan", "renew"],
    },
    "inventory": {
        "objects": ["sku", "warehouse", "purchase_order", "shipment", "stock_level",
                    "supplier", "return_authorization"],
        "verbs": ["create", "receive", "adjust", "list", "cancel", "transfer", "count"],
    },
}

_PARAM_POOL = {
    "id": _STR, "name": _STR, "status": _STR, "limit": _INT, "cursor": _STR,
    "start_date": _STR, "end_date": _STR, "dry_run": _BOOL, "region": _STR,
    "owner": _STR, "reason": _STR, "amount_cents": _INT, "tags": {"type": "array", "items": _STR},
}


def generate_synthetic_tools(count: int = config.SYNTHETIC_TOOL_COUNT,
                             seed: int = config.SYNTHETIC_SEED) -> list[Tool]:
    rng = random.Random(seed)
    combos: list[tuple[str, str, str]] = []
    for domain, spec in _DOMAINS.items():
        for obj in spec["objects"]:
            for verb in spec["verbs"]:
                combos.append((domain, verb, obj))
    rng.shuffle(combos)

    tools: list[Tool] = []
    seen: set[str] = set()
    for domain, verb, obj in combos:
        if len(tools) >= count:
            break
        name = f"{domain}.{verb}_{obj}"
        if name in seen:
            continue
        seen.add(name)
        human_obj = obj.replace("_", " ")
        desc = {
            "create": f"Create a new {human_obj} in the {domain} system.",
            "list": f"List {human_obj} records from the {domain} system with optional filters.",
            "update": f"Update fields on an existing {human_obj} in the {domain} system.",
            "export": f"Export {human_obj} data from the {domain} system as a downloadable file.",
        }.get(verb, f"{verb.capitalize()} the specified {human_obj} in the {domain} system.")
        n_params = rng.randint(1, 4)
        pkeys = rng.sample(list(_PARAM_POOL), n_params)
        params = {"type": "object",
                  "properties": {k: dict(_PARAM_POOL[k]) for k in pkeys},
                  "required": pkeys[:1]}
        tools.append(Tool(name, desc, params, f"synthetic:{domain}", "synthetic"))
    return tools


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


@dataclass
class ToolCatalog:
    tools: list[Tool] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.tools]

    def by_name(self, name: str) -> Tool | None:
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def counts_by_origin(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.tools:
            out[t.origin] = out.get(t.origin, 0) + 1
        return out

    def subset(self, size: int, must_include: Iterable[str] = (), seed: int = 0) -> "ToolCatalog":
        """A deterministic sub-catalog of ``size`` tools that always contains
        ``must_include`` (used by the crossover experiment)."""
        must = [self.by_name(n) for n in must_include]
        must = [t for t in must if t is not None]
        pool = [t for t in self.tools if t.name not in {m.name for m in must}]
        rng = random.Random(seed)
        rng.shuffle(pool)
        chosen = must + pool[: max(0, size - len(must))]
        chosen = chosen[:size] if size >= len(must) else must
        return ToolCatalog(tools=chosen)


def build_catalog(include_real: bool = True, allow_network: bool = True) -> ToolCatalog:
    tools: list[Tool] = []
    if include_real:
        tools.extend(load_real_tools(allow_network=allow_network))
    tools.extend(CURATED_TOOLS)
    tools.extend(generate_synthetic_tools())
    # De-dup by name, keeping first (real > curated > synthetic by insertion order).
    seen: set[str] = set()
    deduped: list[Tool] = []
    for t in tools:
        if t.name in seen:
            continue
        seen.add(t.name)
        deduped.append(t)
    return ToolCatalog(tools=deduped)


if __name__ == "__main__":
    cat = build_catalog(include_real=True, allow_network=True)
    print(f"\nTotal tools: {len(cat.tools)}")
    print("By origin:", json.dumps(cat.counts_by_origin(), indent=2))
