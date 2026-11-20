
import os
import ast
import re
import json
import operator

from pathlib import Path
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ==========================================
# 1. APPLICATION SETUP
# ==========================================

load_dotenv()

app = Flask(__name__)

OUTPUT_DIR = Path("taskpilot_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. PYDANTIC MODELS
# ==========================================

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


# ==========================================
# 3. LANGGRAPH STATE
# ==========================================

class AgentState(TypedDict, total=False):
    user_request: str
    plan: TaskPlan
    messages: Annotated[list[BaseMessage], add_messages]
    review: str


# ==========================================
# 4. GROQ LLM
# ==========================================

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from .env")

    return ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), temperature=0, api_key=api_key)


# ==========================================
# 5. PLAN GENERATOR
# ==========================================

def generate_plan(user_request):

    llm = get_llm()

    prompt = [
        SystemMessage(content="""
            You are TaskPilot Planner.

            Create a practical task plan.

            Return valid JSON only.

            Use this exact structure:

            {
                "goal": "Task goal",
                "assumptions": ["Assumption 1"],
                "steps": [
                    {
                        "step_number": 1,
                        "title": "Step title",
                        "description": "Step description",
                        "estimated_minutes": 5,
                        "expected_output": "Expected result"
                    }
                ],
                "total_estimated_minutes": 10,
                "success_criteria": ["Success criteria"]
            }

            Every step must be a JSON object.
            Never return steps as plain strings.
            Do not include Markdown.
        """),
        HumanMessage(content=user_request)
    ]

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)

    content = str(content).strip()

    # Extract JSON from response

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("Model did not return valid JSON")

    data = json.loads(content[start:end + 1])

    # Convert string steps into dictionaries

    steps = []

    for index, step in enumerate(data.get("steps", []), start=1):

        if isinstance(step, str):
            step = {
                "step_number": index,
                "title": f"Step {index}",
                "description": step,
                "estimated_minutes": 5,
                "expected_output": "Step completed"
            }

        if isinstance(step, dict):
            steps.append({
                "step_number": index,
                "title": str(step.get("title", f"Step {index}")),
                "description": str(step.get("description", "")),
                "estimated_minutes": max(1, int(step.get("estimated_minutes", 5))),
                "expected_output": str(step.get("expected_output", "Step completed"))
            })

    if not steps:
        steps = [{
            "step_number": 1,
            "title": "Complete task",
            "description": user_request,
            "estimated_minutes": 5,
            "expected_output": "Task completed"
        }]

    data["steps"] = steps
    data.setdefault("goal", user_request)
    data.setdefault("assumptions", [])
    data.setdefault("success_criteria", ["Complete the requested task"])
    data["total_estimated_minutes"] = max(1, int(data.get("total_estimated_minutes") or sum(step["estimated_minutes"] for step in steps)))

    return TaskPlan.model_validate(data)


# ==========================================
# 6. CALCULATOR TOOL
# ==========================================

@tool
def calculator(expression: str) -> str:
    """Calculate basic mathematical expressions."""

    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg
    }

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


# ==========================================
# 7. TEXT ANALYZER TOOL
# ==========================================

@tool
def text_analyzer(text: str) -> str:
    """Count words, characters and lines."""

    words = len(text.split())
    characters = len(text)
    lines = len(text.splitlines()) or 1

    return f"Words: {words}, Characters: {characters}, Lines: {lines}"


# ==========================================
# 8. SAVE NOTE TOOL
# ==========================================

@tool
def save_note(filename: str, content: str) -> str:
    """Save text content in a file."""

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


# ==========================================
# 9. REGISTER TOOLS
# ==========================================

TOOLS = [calculator, text_analyzer, save_note]


# ==========================================
# 10. CALCULATION REQUEST DETECTION
# ==========================================

def get_calculation_call(user_request):

    text = user_request.lower()

    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", text)
    numbers = [float(number.replace(",", "")) for number in numbers]

    if len(numbers) < 2:
        return None

    if not any(word in text for word in ["calculate", "calculation", "total cost", "multiply", "multiplication"]):
        return None

    if "each" in text or "total cost" in text or "multiply" in text:
        expression = f"{numbers[0]} * {numbers[1]}"

    elif "divide" in text:
        expression = f"{numbers[0]} / {numbers[1]}"

    elif "subtract" in text:
        expression = f"{numbers[0]} - {numbers[1]}"

    elif "add" in text:
        expression = f"{numbers[0]} + {numbers[1]}"

    else:
        return None

    return expression


# ==========================================
# 11. BUILD LANGGRAPH
# ==========================================

def build_graph():

    llm = get_llm()
    tool_llm = llm.bind_tools(TOOLS, tool_choice="auto")

    # PLANNER NODE

    def planner_node(state: AgentState):
        plan = generate_plan(state["user_request"])
        return {"plan": plan}

    # EXECUTOR NODE

    def executor_node(state: AgentState):

        messages = state.get("messages", [])

        if not messages:

            messages = [
                SystemMessage(content="""
                    You are TaskPilot Executor.

                    Available tools:
                    1. calculator
                    2. text_analyzer
                    3. save_note

                    Use calculator for calculations.
                    Use text_analyzer for text analysis.
                    Use save_note for saving files.

                    Use tools whenever required.
                    Never invent tool results.
                    Give a clear final answer.
                """),
                HumanMessage(content=f"""
                    User Request: {state['user_request']}

                    Task Plan:
                    {state['plan'].model_dump_json(indent=2)}

                    Execute the requested task.
                """)
            ]

            expression = get_calculation_call(state["user_request"])

            if expression:

                tool_call = AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "calculator",
                        "args": {"expression": expression},
                        "id": "calculator_1",
                        "type": "tool_call"
                    }]
                )

                return {"messages": messages + [tool_call]}

        response = tool_llm.invoke(messages)

        return {"messages": [response]}

    # REVIEWER NODE

    def reviewer_node(state: AgentState):

        transcript = "\n".join(f"{message.type}: {message.content}" for message in state.get("messages", []))

        review = llm.invoke([
            SystemMessage(content="You are TaskPilot Reviewer. Review the execution. Report completed tasks, actual tool outputs and limitations. Never invent results."),
            HumanMessage(content=f"""
                User Request: {state['user_request']}

                Task Plan:
                {state['plan'].model_dump_json()}

                Execution:
                {transcript}

                Prepare a final summary.
            """)
        ])

        return {"review": str(review.content)}

    # GRAPH

    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")

    graph.add_conditional_edges("executor", tools_condition, {"tools": "tools", END: "reviewer"})

    graph.add_edge("tools", "executor")
    graph.add_edge("reviewer", END)

    return graph.compile()


# ==========================================
# 12. HOME ROUTE
# ==========================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# ==========================================
# 13. HEALTH CHECK
# ==========================================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "success", "message": "TaskPilot is running"})


# ==========================================
# 14. GENERATE PLAN API
# ==========================================

@app.route("/api/plan", methods=["POST"])
def create_plan():

    try:
        data = request.get_json(silent=True) or {}
        user_request = (data.get("request") or "").strip()

        if len(user_request) < 3:
            return jsonify({"error": "Enter a valid task."}), 400

        plan = generate_plan(user_request)

        return jsonify({"status": "success", "plan": plan.model_dump()})

    except Exception as exc:
        app.logger.exception("Plan generation failed")
        return jsonify({"error": str(exc)}), 500


# ==========================================
# 15. RUN AGENT API
# ==========================================

@app.route("/api/run", methods=["POST"])
def run_agent():

    try:
        data = request.get_json(silent=True) or {}
        user_request = (data.get("request") or "").strip()

        if len(user_request) < 3:
            return jsonify({"error": "Enter a valid task."}), 400

        agent = build_graph()

        result = agent.invoke(
            {"user_request": user_request, "messages": []},
            config={"recursion_limit": 15}
        )

        tool_outputs = [
            {"tool": message.name, "output": str(message.content)}
            for message in result.get("messages", [])
            if isinstance(message, ToolMessage)
        ]

        return jsonify({
            "status": "success",
            "plan": result["plan"].model_dump(),
            "executed_tools": [item["tool"] for item in tool_outputs],
            "tool_outputs": tool_outputs,
            "review": result.get("review", "")
        })

    except Exception as exc:
        app.logger.exception("Agent execution failed")
        return jsonify({"error": str(exc)}), 500


# ==========================================
# 16. RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
