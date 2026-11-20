
import os
import re
import ast
import json
import operator
import uuid

from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# 1. APPLICATION SETUP

load_dotenv()
app = Flask(__name__)
OUTPUT_DIR = Path("taskpilot_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
memory = MemorySaver()


# 2. PYDANTIC MODELS

class TaskStep(BaseModel):
    step_number: int
    title: str
    description: str
    estimated_minutes: int = Field(ge=1)
    expected_output: str


class TaskPlan(BaseModel):
    goal: str
    assumptions: list[str] = Field(default_factory=list)
    steps: list[TaskStep]
    total_estimated_minutes: int = Field(ge=1)
    success_criteria: list[str] = Field(default_factory=list)


# 3. LANGGRAPH STATE

class AgentState(TypedDict, total=False):
    user_request: str
    plan: dict
    approved: bool
    tool_outputs: list
    review: str
    status: str


# 4. GROQ LLM

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from .env")
    return ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), temperature=0, api_key=api_key)


# 5. PLAN GENERATOR

def generate_plan(user_request):
    llm = get_llm()
    planner = llm.bind(response_format={"type": "json_object"})
    prompt = [
        SystemMessage(content=f"""
            You are TaskPilot Planner.
            Create a practical and detailed task plan.
            Return valid JSON only.
            Follow this schema:
            {json.dumps(TaskPlan.model_json_schema())}
            Every step must be a JSON object.
            Never return Markdown or plain string steps.
            Do not execute tasks.
        """),
        HumanMessage(content=user_request)
    ]
    response = planner.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        content = "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
    content = str(content).strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Planner did not return valid JSON.")
    data = json.loads(content[start:end + 1])
    steps = []
    for index, step in enumerate(data.get("steps", []), start=1):
        if isinstance(step, str):
            step = {"title": f"Step {index}", "description": step}
        if isinstance(step, dict):
            steps.append({
                "step_number": index,
                "title": str(step.get("title", f"Step {index}")),
                "description": str(step.get("description", "")),
                "estimated_minutes": max(1, int(step.get("estimated_minutes", 5))),
                "expected_output": str(step.get("expected_output", "Step completed"))
            })
    if not steps:
        steps = [{"step_number": 1, "title": "Complete task", "description": user_request, "estimated_minutes": 5, "expected_output": "Task completed"}]
    data["steps"] = steps
    data["goal"] = str(data.get("goal", user_request))
    data["assumptions"] = data.get("assumptions", [])
    data["success_criteria"] = data.get("success_criteria", ["Complete the requested task"])
    data["total_estimated_minutes"] = max(1, int(data.get("total_estimated_minutes") or sum(step["estimated_minutes"] for step in steps)))
    return TaskPlan.model_validate(data).model_dump()


# 6. CALCULATOR TOOL

@tool
def calculator(expression: str) -> str:
    """Calculate basic mathematical expressions."""
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}
    def calculate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            left = calculate(node.left)
            right = calculate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ValueError("Exponent too large")
            result = operations[type(node.op)](left, right)
            if abs(result) > 1e100:
                raise ValueError("Result too large")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in operations:
            return operations[type(node.op)](calculate(node.operand))
        raise ValueError("Invalid mathematical expression")
    try:
        if len(expression) > 200:
            raise ValueError("Expression too long")
        return str(calculate(ast.parse(expression, mode="eval").body))
    except Exception as exc:
        return f"Calculation error: {exc}"


# 7. TEXT ANALYZER TOOL

@tool
def text_analyzer(text: str) -> str:
    """Count words, characters and lines in text."""
    words = len(text.split())
    characters = len(text)
    lines = len(text.splitlines()) or 1
    return f"Words: {words}, Characters: {characters}, Lines: {lines}"


# 8. SAVE NOTE TOOL

@tool
def save_note(filename: str, content: str) -> str:
    """Save a text report in the taskpilot_outputs folder."""
    name = Path(filename).name
    if not name or name in {".", ".."}:
        return "Invalid filename"
    if not name.endswith(".txt"):
        name += ".txt"
    folder = OUTPUT_DIR.resolve()
    file_path = (folder / name).resolve()
    if file_path.parent != folder:
        return "Invalid file path"
    try:
        file_path.write_text(content, encoding="utf-8")
        return f"File saved successfully: {name}"
    except Exception as exc:
        return f"File saving error: {exc}"


# 9. REGISTER TOOLS

TOOLS = [calculator, text_analyzer, save_note]


# 10. EXECUTE TOOLS

def execute_tools(user_request, plan):
    outputs = []
    text = user_request.lower()
    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", user_request)

    # Detect student/laptop quantity and unit price
    patterns = [
        r"(\d[\d,]*)\s+(?:students|participants|people|laptops|computers|items).*?(?:pay(?:ing)?|cost(?:s|ing)?|priced at|each).*?(\d[\d,]*)",
        r"(\d[\d,]*)\s+(?:students|participants|people|laptops|computers|items).*?each.*?(\d[\d,]*)"
    ]

    calculation_done = False

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            expression = f"{match.group(1).replace(',', '')} * {match.group(2).replace(',', '')}"
            result = calculator.invoke({"expression": expression})
            outputs.append({"tool": "calculator", "input": expression, "output": result})
            calculation_done = True
            break

    # Detect training hours calculation
    hours_match = re.search(r"(\d+)[ -]?day.*?(\d+)\s*hours?\s*(?:each day|per day)", text)

    if hours_match:
        expression = f"{hours_match.group(1)} * {hours_match.group(2)}"
        result = calculator.invoke({"expression": expression})
        outputs.append({"tool": "calculator", "input": expression, "output": result})
        calculation_done = True

    # Simple arithmetic fallback
    if not calculation_done and len(numbers) >= 2 and any(word in text for word in ["calculate", "total cost", "total revenue", "multiply"]):
        expression = f"{numbers[0].replace(',', '')} * {numbers[1].replace(',', '')}"
        result = calculator.invoke({"expression": expression})
        outputs.append({"tool": "calculator", "input": expression, "output": result})

    # Generate report content
    report = f"TaskPilot Report\n\nUser Request:\n{user_request}\n\nTask Plan:\n{json.dumps(plan, indent=2)}\n\nCalculation Results:\n{json.dumps(outputs, indent=2)}"

    # Text analyzer
    if any(word in text for word in ["analyze", "analyse", "word count", "count words", "text analysis"]):
        result = text_analyzer.invoke({"text": report})
        outputs.append({"tool": "text_analyzer", "input": "Generated task report", "output": result})

    # Save report
    if any(word in text for word in ["save", "create a file", "save it"]):
        match = re.search(r"([a-zA-Z0-9_-]+\.txt)", user_request)
        filename = match.group(1) if match else "taskpilot_report.txt"
        report += f"\n\nAll Tool Outputs:\n{json.dumps(outputs, indent=2)}"
        result = save_note.invoke({"filename": filename, "content": report})
        outputs.append({"tool": "save_note", "input": filename, "output": result})

    if not outputs:
        outputs.append({"tool": "task_planner", "input": user_request, "output": "Plan created. No matching execution tool was requested."})

    return outputs


# 11. BUILD LANGGRAPH

def build_graph():
    llm = get_llm()

    # PLANNER NODE
    def planner_node(state: AgentState):
        plan = generate_plan(state["user_request"])
        return {"plan": plan, "status": "plan_created"}

    # HUMAN APPROVAL NODE
    def approval_node(state: AgentState):
        decision = interrupt({
            "message": "TaskPilot has generated a task plan.",
            "question": "Do you approve this plan?",
            "plan": state["plan"]
        })
        return {"approved": decision.get("approved", False)}

    # APPROVAL ROUTER
    def approval_router(state: AgentState):
        return "executor" if state.get("approved") is True else "rejected"

    # EXECUTOR NODE
    def executor_node(state: AgentState):
        outputs = execute_tools(state["user_request"], state["plan"])
        return {"tool_outputs": outputs, "status": "executed"}

    # REVIEWER NODE
    def reviewer_node(state: AgentState):
        try:
            response = llm.invoke([
                SystemMessage(content="You are TaskPilot Reviewer. Review the actual execution results. Explain completed work, tools used, results and limitations. Never invent results."),
                HumanMessage(content=f"User Request: {state['user_request']}\nPlan: {json.dumps(state['plan'], indent=2)}\nActual Tool Outputs: {json.dumps(state.get('tool_outputs', []), indent=2)}")
            ])
            review = str(response.content)
        except Exception as exc:
            review = f"Execution finished. Review generation failed: {exc}"
        return {"review": review, "status": "completed"}

    # REJECT NODE
    def rejected_node(state: AgentState):
        return {"status": "rejected", "review": "Plan rejected. No tools were executed."}

    # GRAPH
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("approval", approval_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("rejected", rejected_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "approval")
    graph.add_conditional_edges("approval", approval_router, {"executor": "executor", "rejected": "rejected"})
    graph.add_edge("executor", "reviewer")
    graph.add_edge("reviewer", END)
    graph.add_edge("rejected", END)
    return graph.compile(checkpointer=memory)


# 12. HOME PAGE

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# 13. HEALTH CHECK

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "success", "message": "TaskPilot is running"})


# 14. RUN AGENT API

@app.route("/api/run", methods=["POST"])
def run_agent():
    try:
        data = request.get_json(silent=True) or {}
        user_request = (data.get("request") or "").strip()

        if len(user_request) < 3:
            return jsonify({"error": "Enter a valid task."}), 400

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}
        agent = build_graph()
        result = agent.invoke({"user_request": user_request}, config=config)
        pending = result.get("__interrupt__", [])

        if pending:
            return jsonify({
                "status": "awaiting_approval",
                "thread_id": thread_id,
                "approval_request": pending[0].value
            })

        return jsonify({"status": result.get("status"), "review": result.get("review", "")})

    except Exception as exc:
        app.logger.exception("Agent execution failed")
        return jsonify({"error": str(exc)}), 500


# 15. APPROVAL API

@app.route("/api/approve", methods=["POST"])
def approve_agent():
    try:
        data = request.get_json(silent=True) or {}
        thread_id = data.get("thread_id")
        approved = data.get("approved")

        if not thread_id or not isinstance(approved, bool):
            return jsonify({"error": "Invalid approval request."}), 400

        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}
        agent = build_graph()
        result = agent.invoke(Command(resume={"approved": approved}), config=config)

        return jsonify({
            "status": result.get("status"),
            "review": result.get("review", ""),
            "tool_outputs": result.get("tool_outputs", [])
        })

    except Exception as exc:
        app.logger.exception("Approval execution failed")
        return jsonify({"error": str(exc)}), 500


# 16. GENERATE PLAN API

@app.route("/api/plan", methods=["POST"])
def create_plan():
    try:
        data = request.get_json(silent=True) or {}
        user_request = (data.get("request") or "").strip()

        if len(user_request) < 3:
            return jsonify({"error": "Enter a valid task."}), 400

        plan = generate_plan(user_request)
        return jsonify({"status": "success", "plan": plan})

    except Exception as exc:
        app.logger.exception("Plan generation failed")
        return jsonify({"error": str(exc)}), 500


# 17. RUN FLASK

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

