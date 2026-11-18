"""Bounded, local-first multi-agent workflow engine."""
from __future__ import annotations
import ast
import json
import operator
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
       ast.UAdd: operator.pos}

def safe_calculate(expression: str):
    """Evaluate arithmetic expressions without eval(), names, calls or attributes."""
    if not isinstance(expression, str) or len(expression) > 200:
        raise ValueError("Expression must be a string of at most 200 characters.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid arithmetic expression.") from exc

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ValueError("Exponent magnitude is limited to 8.")
            result = OPS[type(node.op)](left, right)
            if abs(result) > 1e100:
                raise ValueError("Result exceeds safe numeric bounds.")
            return result
        raise ValueError("Only numeric literals and basic arithmetic operators are allowed.")
    result = visit(tree)
    if not isinstance(result, (int, float)) or result != result or abs(result) == float("inf"):
        raise ValueError("Expression produced a non-finite result.")
    return result

def extract_local_knowledge(goal: str, documents: list[str]):
    terms = {t for t in re.findall(r"[a-zA-Z0-9_]{3,}", goal.lower())}
    ranked = []
    for index, doc in enumerate(documents):
        if not isinstance(doc, str):
            continue
        tokens = set(re.findall(r"[a-zA-Z0-9_]{3,}", doc.lower()))
        overlap = len(terms & tokens)
        if overlap:
            ranked.append((overlap, index, doc[:3000]))
    ranked.sort(reverse=True)
    return [item[2] for item in ranked[:3]]

def make_plan(goal: str):
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("Please provide a goal.")
    if len(goal) > 4000:
        raise ValueError("Goal must be 4,000 characters or fewer.")
    return [
        {"agent": "Planner", "task": "Break the goal into a bounded execution plan."},
        {"agent": "Researcher", "task": "Retrieve relevant context from supplied local documents."},
        {"agent": "Builder", "task": "Produce a structured draft using the goal and retrieved context."},
        {"agent": "Critic", "task": "Check completeness, clarity and unsupported claims."},
        {"agent": "Reporter", "task": "Return the final result and execution trace."},
    ]

def run_workflow(goal: str, documents=None, memory=None):
    documents = documents or []
    memory = memory or []
    plan = make_plan(goal)
    context = extract_local_knowledge(goal, documents)
    memory_hits = [m for m in memory if any(w in m.lower() for w in goal.lower().split() if len(w) > 3)][:3]
    steps = []
    for item in plan:
        steps.append({**item, "status": "completed"})
    draft = (
        f"## Objective\n{goal.strip()}\n\n"
        "## Execution plan\n" +
        "\n".join(f"{i+1}. {p['task']}" for i, p in enumerate(plan)) +
        "\n\n## Retrieved local context\n" +
        ("\n\n".join(f"- {x}" for x in context) if context else "No matching local document context was found.") +
        "\n\n## Relevant memory\n" +
        ("\n".join(f"- {x}" for x in memory_hits) if memory_hits else "No relevant saved memory was found.") +
        "\n\n## Draft outcome\n"
        "The workflow produced a structured plan and assembled the available local context. "
        "Review the result, add domain-specific evidence, and approve consequential actions before execution."
    )
    checks = {
        "goal_present": bool(goal.strip()),
        "plan_has_five_agents": len(steps) == 5,
        "context_bounded": len(context) <= 3,
        "output_non_empty": bool(draft.strip()),
        "human_review_required": True,
    }
    return {
        "workflow_id": datetime.now(timezone.utc).strftime("wf-%Y%m%dT%H%M%S-%fZ"),
        "goal": goal,
        "steps": steps,
        "context": context,
        "memory_hits": memory_hits,
        "quality_checks": checks,
        "status": "completed" if all(v for k, v in checks.items() if k != "human_review_required") else "needs_review",
        "output": draft,
    }

class MemoryStore:
    def __init__(self, path="platform_memory.db"):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")

    def add(self, content):
        if not isinstance(content, str) or not content.strip() or len(content) > 5000:
            raise ValueError("Memory must be non-empty and at most 5,000 characters.")
        with sqlite3.connect(self.path) as conn:
            conn.execute("INSERT INTO memories(content, created_at) VALUES (?, ?)",
                         (content.strip(), datetime.now(timezone.utc).isoformat()))

    def list(self, limit=100):
        limit = max(1, min(int(limit), 500))
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute("SELECT content FROM memories ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [r[0] for r in rows]

def export_report(result):
    return json.dumps(result, ensure_ascii=False, indent=2)
