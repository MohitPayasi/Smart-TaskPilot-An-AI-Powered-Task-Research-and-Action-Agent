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
from pydantic import BaseModel, Field, ValidationError

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

app = Flask(__name__)
OUTPUT_DIR = Path("taskpilot_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# One shared checkpointer is essential: /api/run and /api/approve
# must access the same saved graph state.
checkpointer = MemorySaver()


def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=GROQ_API_KEY,
    )


# ============================================================
# 2. PYDANTIC OUTPUT MODELS
# ============================================================

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


# ============================================================
# 3. SAFE JSON HELPERS
# ============================================================

def response_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def parse_json_object(content: Any) -> dict:
    raw = response_text(content).strip()
    # Remove optional Markdown fences if the model adds them.
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Model did not return a JSON object: {raw[:500]}")
    value = json.loads(raw[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


# ============================================================
# 4. PLANNER
# ============================================================

def generate_plan(user_request: str) -> dict:
    llm = get_llm().bind(response_format={"type": "json_object"})

    messages = [
        SystemMessage(content=f"""
You are TaskPilot's planning agent.
Create a useful, realistic plan for the user's request.
The request can be a question, calculation, explanation, analysis,
learning plan, workshop, writing task, or file-related task.

Return ONLY a JSON object matching this schema:
{json.dumps(TaskPlan.model_json_schema(), ensure_ascii=False)}

Rules:
- Make steps specific to the user's actual request.
- Do not claim that any step has already been executed.
- For a simple question, use a short plan.
- For a complex task, provide enough detail to make the plan actionable.
- Do not invent missing user-specific facts. Put necessary assumptions in assumptions.
- Estimated minutes must be positive integers.
- Every step must be an object.
"""),
        HumanMessage(content=user_request),
    ]

    data = parse_json_object(llm.invoke(messages).content)

    # Normalize common model omissions before strict Pydantic validation.
    normalized_steps = []
    for i, step in enumerate(data.get("steps", []), start=1):
        if isinstance(step, str):
            step = {"title": f"Step {i}", "description": step}
        if not isinstance(step, dict):
            continue
        normalized_steps.append({
            "step_number": i,
            "title": str(step.get("title") or f"Step {i}"),
            "description": str(step.get("description") or ""),
            "estimated_minutes": max(1, int(step.get("estimated_minutes") or 5)),
            "expected_output": str(step.get("expected_output") or "Step result"),
        })

    if not normalized_steps:
        normalized_steps = [{
            "step_number": 1,
            "title": "Prepare the response",
            "description": user_request,
            "estimated_minutes": 5,
            "expected_output": "A response addressing the user's request",
        }]

    total = sum(s["estimated_minutes"] for s in normalized_steps)
    normalized = {
        "goal": str(data.get("goal") or user_request),
        "assumptions": data.get("assumptions") or [],
        "steps": normalized_steps,
        "total_estimated_minutes": max(
            1, int(data.get("total_estimated_minutes") or total)
        ),
        "success_criteria": data.get("success_criteria") or [
            "The result addresses the user's original request."
        ],
    }

    return TaskPlan.model_validate(normalized).model_dump()


# ============================================================
# 5. TOOLS
# ============================================================

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression. Supports +, -, *, /, ** and parentheses."""
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Mod: operator.mod,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ValueError("Exponent is too large.")
            result = allowed_ops[type(node.op)](left, right)
            if not isinstance(result, (int, float)) or abs(result) > 1e100:
                raise ValueError("Result is outside the allowed range.")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](evaluate(node.operand))
        raise ValueError("Unsupported expression.")

    if len(expression) > 200:
        return "Error: expression is too long."
    try:
        tree = ast.parse(expression, mode="eval")
        return str(evaluate(tree.body))
    except Exception as exc:
        return f"Calculation error: {exc}"


@tool
def text_analyzer(text: str) -> str:
    """Analyze supplied text and return word, character, and line counts."""
    return json.dumps({
        "words": len(text.split()),
        "characters": len(text),
        "lines": len(text.splitlines()) or 1,
    })


@tool
def save_note(filename: str, content: str) -> str:
    """Save user-requested text content as a .txt file in taskpilot_outputs."""
    safe_name = Path(filename).name
    if not safe_name or safe_name in {".", ".."}:
        return "Error: invalid filename."
    if not safe_name.lower().endswith(".txt"):
        safe_name += ".txt"

    destination = (OUTPUT_DIR.resolve() / safe_name).resolve()
    if destination.parent != OUTPUT_DIR.resolve():
        return "Error: invalid output path."

    try:
        destination.write_text(content, encoding="utf-8")
        return f"Saved successfully: {destination.name}"
    except OSError as exc:
        return f"File save error: {exc}"


TOOLS = [calculator, text_analyzer, save_note]
TOOLS_BY_NAME = {item.name: item for item in TOOLS}


# ============================================================
# 6. TOOL-CALLING EXECUTOR
# ============================================================

def execute_request(user_request: str, plan: dict) -> list[dict]:
    """
    The model decides whether a tool is needed.
    If no tool is needed, it can answer normally.
    Tool execution is limited to the explicitly registered tools.
    """
    llm = get_llm().bind_tools(TOOLS)
    messages = [
        SystemMessage(content="""
You are TaskPilot's execution agent.

Help complete the user's original request using the approved plan.
You may use only the tools provided to you:
- calculator: arithmetic only
- text_analyzer: analyze text supplied in the request
- save_note: save content only when the user explicitly requests a file/note

Important:
- Do not call a tool just to appear busy.
- Use calculator for calculations instead of mental arithmetic.
- Use text_analyzer only when the user asks to analyze/count supplied text.
- Use save_note only when the user asks to save/create a text file.
- Do not claim to create Excel, PDF, Word, browse the web, send emails,
  or perform external actions because no such tool is registered.
- If the user asks a normal knowledge or writing question, answer it directly.
- Follow the approved plan.
"""),
        HumanMessage(content=(
            f"USER REQUEST:\n{user_request}\n\n"
            f"APPROVED PLAN:\n{json.dumps(plan, ensure_ascii=False)}"
        )),
    ]

    outputs: list[dict] = []

    # Bounded tool loop prevents endless tool-calling.
    for _ in range(6):
        ai_message = llm.invoke(messages)
        messages.append(ai_message)

        calls = getattr(ai_message, "tool_calls", None) or []
        if not calls:
            outputs.append({
                "type": "assistant_response",
                "output": response_text(ai_message.content),
            })
            break

        for call in calls:
            name = call.get("name")
            args = call.get("args") or {}
            call_id = call.get("id")

            if name not in TOOLS_BY_NAME:
                result = f"Tool '{name}' is not available."
            else:
                try:
                    result = TOOLS_BY_NAME[name].invoke(args)
                except Exception as exc:
                    result = f"Tool execution error: {exc}"

            outputs.append({
                "type": "tool",
                "tool": name,
                "input": args,
                "output": str(result),
            })
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=call_id,
            ))
    else:
        outputs.append({
            "type": "system",
            "output": "Stopped after reaching the maximum tool-call limit.",
        })

    return outputs


# ============================================================
# 7. FINAL STRUCTURED REPORT
# ============================================================

def create_final_report(user_request: str, plan: dict,
                        tool_outputs: list[dict]) -> dict:
    llm = get_llm().bind(response_format={"type": "json_object"})

    messages = [
        SystemMessage(content=f"""
You are TaskPilot's final response formatter and reviewer.

Return ONLY valid JSON matching this schema:
{json.dumps(FinalReport.model_json_schema(), ensure_ascii=False)}

Your response MUST contain the complete answer inside the tables array.
Do not put the substantive answer in prose outside the tables.

Rules:
- Answer the user's ORIGINAL request, not merely describe the plan.
- Use the approved plan as guidance and actual tool outputs when relevant.
- Never invent a tool result, calculation, file, or completed action.
- If an assistant_response exists, use it as the main factual source.
- If no tool was needed, answer directly using tables.
- Create useful sections such as "Request Summary", "Detailed Answer",
  "Execution Results", "Success Criteria", and "Limitations" as relevant.
- Every table must have a section, headers, and rows.
- Every row must have the same number of cells as its table headers.
- Keep cells concise but sufficiently informative.
- Mention unavailable capabilities in a Limitations table when relevant.
- Do not claim success for an action that was not executed.
- status must be one of: completed, partially_completed, failed.
"""),
        HumanMessage(content=(
            f"ORIGINAL USER REQUEST:\n{user_request}\n\n"
            f"APPROVED PLAN:\n{json.dumps(plan, ensure_ascii=False)}\n\n"
            f"ACTUAL EXECUTION OUTPUTS:\n{json.dumps(tool_outputs, ensure_ascii=False)}"
        )),
    ]

    data = parse_json_object(llm.invoke(messages).content)

    # Normalize rows so every row matches its table's header count.
    normalized_tables = []
    for table in data.get("tables", []):
        if not isinstance(table, dict):
            continue
        headers = [str(value) for value in (table.get("headers") or [])]
        if not headers:
            headers = ["Details"]
        rows = []
        for row in table.get("rows", []):
            if isinstance(row, (list, tuple)):
                cells = [str(value) for value in row]
            else:
                cells = [str(row)]
            cells = cells[:len(headers)]
            cells.extend([""] * (len(headers) - len(cells)))
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
            "rows": [["Answer", "The model did not return a structured table."]],
        }]

    normalized = {
        "title": str(data.get("title") or "TaskPilot Final Review"),
        "status": str(data.get("status") or "completed"),
        "tables": normalized_tables,
    }
    if normalized["status"] not in {"completed", "partially_completed", "failed"}:
        normalized["status"] = "partially_completed"

    return FinalReport.model_validate(normalized).model_dump()


# ============================================================
# 8. LANGGRAPH WITH MANDATORY HUMAN APPROVAL
# ============================================================

def build_graph():
    def planner_node(state: AgentState):
        plan = generate_plan(state["user_request"])
        return {"plan": plan, "status": "plan_created"}

    def human_approval_node(state: AgentState):
        # Graph pauses here. No executor node runs until resume approval.
        decision = interrupt({
            "message": "TaskPilot has prepared a plan. Review it before execution.",
            "question": "Do you approve this plan?",
            "plan": state["plan"],
            "options": ["Approve", "Reject"],
        })
        return {"approved": bool(decision.get("approved", False))}

    def approval_router(state: AgentState):
        return "executor" if state.get("approved") is True else "rejected"

    def executor_node(state: AgentState):
        outputs = execute_request(state["user_request"], state["plan"])
        return {"tool_outputs": outputs, "status": "executed"}

    def reviewer_node(state: AgentState):
        report = create_final_report(
            state["user_request"],
            state["plan"],
            state.get("tool_outputs", []),
        )
        return {
            "report": report,
            # Keep the full structured report in review as well.
            # FinalReport contains title, status and tables, not final_answer.
            "review": report,
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
        approval_router,
        {"executor": "executor", "rejected": "rejected"},
    )
    workflow.add_edge("executor", "reviewer")
    workflow.add_edge("reviewer", END)
    workflow.add_edge("rejected", END)

    return workflow.compile(checkpointer=checkpointer)


agent = build_graph()


# ============================================================
# 9. FLASK ROUTES
# ============================================================

@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "success",
        "message": "TaskPilot is running",
        "approval_required": True,
    })


@app.post("/api/run")
def run_agent():
    try:
        data = request.get_json(silent=True) or {}
        user_request = str(data.get("request") or "").strip()

        if len(user_request) < 3:
            return jsonify({"error": "Please enter a valid request."}), 400

        thread_id = str(uuid.uuid4())
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 20,
        }

        result = agent.invoke(
            {"user_request": user_request},
            config=config,
        )

        pending = result.get("__interrupt__", [])
        if pending:
            return jsonify({
                "status": "awaiting_approval",
                "thread_id": thread_id,
                "approval_request": pending[0].value,
            }), 200

        # Defensive response: normal flow should pause for approval.
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
        thread_id = data.get("thread_id")
        approved = data.get("approved")

        if not thread_id or not isinstance(approved, bool):
            return jsonify({
                "error": "thread_id and boolean approved are required."
            }), 400

        config = {
            "configurable": {"thread_id": str(thread_id)},
            "recursion_limit": 20,
        }

        result = agent.invoke(
            Command(resume={"approved": approved}),
            config=config,
        )

        # A second interrupt should not normally occur, but handle it safely.
        pending = result.get("__interrupt__", [])
        if pending:
            return jsonify({
                "status": "awaiting_approval",
                "thread_id": thread_id,
                "approval_request": pending[0].value,
            }), 200

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
    """
    Preview-only endpoint. It generates a plan but does not execute anything.
    The /api/run endpoint is the approval-gated execution workflow.
    """
    try:
        data = request.get_json(silent=True) or {}
        user_request = str(data.get("request") or "").strip()

        if len(user_request) < 3:
            return jsonify({"error": "Please enter a valid request."}), 400

        plan = generate_plan(user_request)
        return jsonify({"status": "success", "plan": plan}), 200

    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        app.logger.exception("Plan validation failed")
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:
        app.logger.exception("Plan generation failed")
        return jsonify({"error": str(exc)}), 500


# ============================================================
# 10. START SERVER
# ============================================================

if __name__ == "__main__":
    # Use debug=False outside local development.
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
