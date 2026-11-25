import ast
import json
import operator
import os
import re
import uuid
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

load_dotenv()

app = Flask(__name__)
OUTPUT_DIR = Path("taskpilot_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

checkpointer = MemorySaver()


def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
    return ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)


class TaskStep(BaseModel):
    step_number: int
    title: str
    description: str
    estimated_minutes: int = Field(default=5, ge=1)
    expected_output: str = "Step result"


class TaskPlan(BaseModel):
    goal: str
    assumptions: list[str] = Field(default_factory=list)
    steps: list[TaskStep] = Field(min_length=1)
    total_estimated_minutes: int = Field(default=1, ge=1)
    success_criteria: list[str] = Field(default_factory=list)


class ReportTable(BaseModel):
    section: str
    headers: list[str] = Field(min_length=1)
    rows: list[list[str]] = Field(default_factory=list)


class FinalReport(BaseModel):
    title: str = "TaskPilot Final Review"
    status: str = "completed"
    tables: list[ReportTable] = Field(min_length=1)


class AgentState(TypedDict, total=False):
    user_request: str
    plan: dict
    approved: bool
    tool_outputs: list[dict]
    report: dict
    review: str
    status: str


def parse_json_object(content: Any) -> dict:
    if isinstance(content, list):
        text = "\n".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
    else:
        text = str(content)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model did not return a valid JSON object.")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def generate_plan(user_request: str) -> dict:
    llm = get_llm().bind(response_format={"type": "json_object"})
    messages = [
        SystemMessage(content=f"""You are TaskPilot Planner. Create an actionable task plan for the user's request.
Return ONLY a valid JSON object matching this schema:
{json.dumps(TaskPlan.model_json_schema())}"""),
        HumanMessage(content=user_request),
    ]

    data = parse_json_object(llm.invoke(messages).content)

    steps = []
    for i, step in enumerate(data.get("steps", []), start=1):
        if isinstance(step, str):
            step = {"title": f"Step {i}", "description": step}
        if isinstance(step, dict):
            steps.append({
                "step_number": i,
                "title": str(step.get("title") or f"Step {i}"),
                "description": str(step.get("description") or ""),
                "estimated_minutes": max(1, int(step.get("estimated_minutes") or 5)),
                "expected_output": str(step.get("expected_output") or "Step result"),
            })

    if not steps:
        steps = [{
            "step_number": 1,
            "title": "Prepare response",
            "description": user_request,
            "estimated_minutes": 5,
            "expected_output": "Addressed user request",
        }]

    normalized = {
        "goal": str(data.get("goal") or user_request),
        "assumptions": data.get("assumptions") or [],
        "steps": steps,
        "total_estimated_minutes": max(1, int(data.get("total_estimated_minutes") or sum(s["estimated_minutes"] for s in steps))),
        "success_criteria": data.get("success_criteria") or ["The request is successfully addressed."],
    }
    return TaskPlan.model_validate(normalized).model_dump()


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely (+, -, *, /, **, %)."""
    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            right = _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ValueError("Exponent too large.")
            return ops[type(node.op)](_eval(node.left), right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression.")

    try:
        return str(_eval(ast.parse(expression.strip(), mode="eval").body))
    except Exception as exc:
        return f"Calculation error: {exc}"


@tool
def text_analyzer(text: str) -> str:
    """Analyze text and return word, character, and line counts."""
    return json.dumps({
        "words": len(text.split()),
        "characters": len(text),
        "lines": len(text.splitlines()) or 1,
    })


@tool
def save_note(filename: str, content: str) -> str:
    """Save text content as a .txt file in taskpilot_outputs."""
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".txt"):
        safe_name += ".txt"
    dest = (OUTPUT_DIR.resolve() / safe_name).resolve()
    if dest.parent != OUTPUT_DIR.resolve():
        return "Error: Invalid file destination."
    try:
        dest.write_text(content, encoding="utf-8")
        return f"Saved successfully: {dest.name}"
    except Exception as exc:
        return f"File save error: {exc}"


TOOLS = [calculator, text_analyzer, save_note]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def execute_request(user_request: str, plan: dict) -> list[dict]:
    llm = get_llm().bind_tools(TOOLS)
    messages = [
        SystemMessage(content="""You are TaskPilot's execution agent.
Complete the user request using the approved plan and available tools (calculator, text_analyzer, save_note).
If no tool is required, answer directly."""),
        HumanMessage(content=f"REQUEST: {user_request}\nPLAN: {json.dumps(plan)}"),
    ]

    outputs = []
    for _ in range(6):
        ai_msg = llm.invoke(messages)
        messages.append(ai_msg)
        calls = getattr(ai_msg, "tool_calls", None) or []
        if not calls:
            text = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
            outputs.append({"type": "assistant_response", "output": text})
            break

        for call in calls:
            name, args, call_id = call.get("name"), call.get("args") or {}, call.get("id")
            selected_tool = TOOLS_BY_NAME.get(name)
            result = str(selected_tool.invoke(args)) if selected_tool else f"Tool '{name}' not found."
            outputs.append({"type": "tool", "tool": name, "input": args, "output": result})
            messages.append(ToolMessage(content=result, tool_call_id=call_id))

    return outputs


def create_final_report(user_request: str, plan: dict, tool_outputs: list[dict]) -> dict:
    llm = get_llm().bind(response_format={"type": "json_object"})
    messages = [
        SystemMessage(content=f"""You are TaskPilot Reviewer. Formulate the final answer strictly inside structured tables.
Return valid JSON matching this schema:
{json.dumps(FinalReport.model_json_schema())}
Rules:
- Address the user request thoroughly using actual tool outputs when available.
- Every table must have a section name, headers list, and rows list (where each row length matches headers length)."""),
        HumanMessage(content=f"REQUEST: {user_request}\nPLAN: {json.dumps(plan)}\nTOOL OUTPUTS: {json.dumps(tool_outputs)}"),
    ]

    data = parse_json_object(llm.invoke(messages).content)

    normalized_tables = []
    for table in data.get("tables", []):
        if not isinstance(table, dict):
            continue
        headers = [str(h) for h in (table.get("headers") or ["Details"])]
        rows = []
        for row in table.get("rows", []):
            cells = [str(c) for c in (row if isinstance(row, (list, tuple)) else [row])]
            cells = cells[:len(headers)] + [""] * max(0, len(headers) - len(cells))
            rows.append(cells)
        normalized_tables.append({
            "section": str(table.get("section") or "Report"),
            "headers": headers,
            "rows": rows,
        })

    if not normalized_tables:
        normalized_tables = [{
            "section": "Final Answer",
            "headers": ["Field", "Details"],
            "rows": [["Answer", "No structured tables were returned."]],
        }]

    return FinalReport.model_validate({
        "title": str(data.get("title") or "TaskPilot Final Review"),
        "status": data.get("status") if data.get("status") in {"completed", "partially_completed", "failed"} else "completed",
        "tables": normalized_tables,
    }).model_dump()

def build_graph():
    def planner_node(state: AgentState):
        return {"plan": generate_plan(state["user_request"]), "status": "plan_created"}

    def human_approval_node(state: AgentState):
        decision = interrupt({
            "message": "TaskPilot has prepared a plan. Review it before execution.",
            "question": "Do you approve this plan?",
            "plan": state["plan"],
            "options": ["Approve", "Reject"],
        })
        return {"approved": bool(decision.get("approved", False))}

    def executor_node(state: AgentState):
        return {"tool_outputs": execute_request(state["user_request"], state["plan"]), "status": "executed"}

    def reviewer_node(state: AgentState):
        report = create_final_report(state["user_request"], state["plan"], state.get("tool_outputs", []))
        return {
            "report": report,
            "review": report.get("title", "Review completed"),
            "status": report.get("status", "completed"),
        }

    def rejected_node(state: AgentState):
        return {
            "approved": False,
            "tool_outputs": [],
            "status": "rejected",
            "review": "Plan rejected by the user. No tools were executed.",
            "report": None,
        }

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("rejected", rejected_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "human_approval")
    workflow.add_conditional_edges(
        "human_approval",
        lambda state: "executor" if state.get("approved") else "rejected",
        ["executor", "rejected"],
    )
    workflow.add_edge("executor", "reviewer")
    workflow.add_edge("reviewer", END)
    workflow.add_edge("rejected", END)

    return workflow.compile(checkpointer=checkpointer)


agent = build_graph()



@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "success", "message": "TaskPilot is running", "approval_required": True})


@app.post("/api/run")
def run_agent():
    try:
        user_request = str((request.get_json(silent=True) or {}).get("request") or "").strip()
        if len(user_request) < 3:
            return jsonify({"error": "Please enter a valid request."}), 400

        thread_id = str(uuid.uuid4())
        result = agent.invoke(
            {"user_request": user_request},
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": 20},
        )

        pending = result.get("__interrupt__", [])
        if pending:
            return jsonify({
                "status": "awaiting_approval",
                "thread_id": thread_id,
                "approval_request": pending[0].value,
            }), 200

        return jsonify({
            "status": result.get("status", "unknown"),
            "report": result.get("report"),
            "review": result.get("review", ""),
        }), 200

    except Exception as exc:
        app.logger.exception("TaskPilot run failed")
        return jsonify({"error": str(exc)}), 500


@app.post("/api/approve")
def approve_agent():
    try:
        data = request.get_json(silent=True) or {}
        thread_id, approved = data.get("thread_id"), data.get("approved")
        if not thread_id or not isinstance(approved, bool):
            return jsonify({"error": "thread_id and boolean approved are required."}), 400

        result = agent.invoke(
            Command(resume={"approved": approved}),
            config={"configurable": {"thread_id": str(thread_id)}, "recursion_limit": 20},
        )

        return jsonify({
            "status": result.get("status", "unknown"),
            "approved": approved,
            "report": result.get("report"),
            "review": result.get("review", ""),
            "tool_outputs": result.get("tool_outputs", []),
        }), 200

    except Exception as exc:
        app.logger.exception("TaskPilot approval failed")
        return jsonify({"error": str(exc)}), 500


@app.post("/api/plan")
def create_plan():
    try:
        user_request = str((request.get_json(silent=True) or {}).get("request") or "").strip()
        if len(user_request) < 3:
            return jsonify({"error": "Please enter a valid request."}), 400
        return jsonify({"status": "success", "plan": generate_plan(user_request)}), 200
    except Exception as exc:
        app.logger.exception("Plan generation failed")
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
