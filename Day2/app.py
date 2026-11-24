import os
import ast
import json
import math
import operator
from pathlib import Path
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage,
    AIMessage, ToolMessage
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024

OUTPUT_DIR = BASE_DIR / "taskpilot_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Pydantic models
# --------------------------------------------------

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



class AgentState(TypedDict, total=False):
    user_request: str
    plan: TaskPlan
    messages: Annotated[list[BaseMessage], add_messages]
    review: str


# --------------------------------------------------
# LLM
# --------------------------------------------------

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to the .env file."
        )

    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    return ChatGroq(
        model=model_name,
        temperature=0,
        api_key=api_key
    )


# --------------------------------------------------
# Planner
# --------------------------------------------------

def generate_plan(user_request: str) -> TaskPlan:
    llm = get_llm()

    messages = [
        SystemMessage(content="""
You are TaskPilot Planner.
Create a practical task plan.
Return valid JSON only, with this exact structure:
{
  "goal": "Task goal",
  "assumptions": ["Assumption"],
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
  "success_criteria": ["Success criterion"]
}
Every step must be an object, not a string.
estimated_minutes must be a positive integer.
Do not include Markdown or text outside the JSON.
"""),
        HumanMessage(content=user_request)
    ]

    response = llm.invoke(messages)
    content = response.content

    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict)
            else str(item)
            for item in content
        )

    content = str(content).strip()
    start = content.find("{")
    end = content.rfind("}")

    if start < 0 or end < start:
        raise ValueError("Planner did not return a JSON object.")

    data = json.loads(content[start:end + 1])
    normalized_steps = []

    for index, step in enumerate(data.get("steps", []), start=1):
        if isinstance(step, str):
            step = {
                "title": f"Step {index}",
                "description": step,
                "estimated_minutes": 5,
                "expected_output": "Step completed"
            }

        if not isinstance(step, dict):
            continue

        normalized_steps.append({
            "step_number": index,
            "title": str(step.get("title", f"Step {index}")),
            "description": str(step.get("description", "")),
            "estimated_minutes": max(1, int(step.get("estimated_minutes", 5))),
            "expected_output": str(step.get("expected_output", "Step completed"))
        })

    if not normalized_steps:
        raise ValueError("Planner returned no valid task steps.")

    data["steps"] = normalized_steps
    data.setdefault("goal", user_request)
    data.setdefault("assumptions", [])
    data.setdefault("success_criteria", ["Complete the requested task"])
    data["total_estimated_minutes"] = sum(
        step["estimated_minutes"] for step in normalized_steps
    )

    return TaskPlan.model_validate(data)


# --------------------------------------------------
# Tools
# --------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression using numbers and +, -, *, /, **."""

    allowed_operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in allowed_operations:
            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ValueError("Exponent is too large.")

            result = allowed_operations[type(node.op)](left, right)

            if not math.isfinite(result) or abs(result) > 1e100:
                raise ValueError("Calculation result is outside the allowed range.")

            return result

        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_operations:
            return allowed_operations[type(node.op)](evaluate(node.operand))

        raise ValueError("Invalid expression. Only basic arithmetic is supported.")

    try:
        if len(expression) > 200:
            raise ValueError("Expression is too long.")

        tree = ast.parse(expression, mode="eval")
        return str(evaluate(tree.body))
    except Exception as exc:
        return f"Calculation error: {exc}"


@tool
def text_analyzer(text: str) -> str:
    """Count words, characters, and lines in a text."""
    words = len(text.split())
    characters = len(text)
    lines = len(text.splitlines()) or 1

    return (
        f"Words: {words}, "
        f"Characters: {characters}, "
        f"Lines: {lines}"
    )


@tool
def save_note(filename: str, content: str) -> str:
    """Save text content as a TXT file inside the taskpilot_outputs folder."""
    name = Path(filename).name.strip()

    if not name or name in {".", ".."}:
        return "Invalid filename."

    if not name.lower().endswith(".txt"):
        name += ".txt"

    file_path = (OUTPUT_DIR / name).resolve()

    if file_path.parent != OUTPUT_DIR.resolve():
        return "Invalid file path."

    try:
        file_path.write_text(content, encoding="utf-8")
        return f"File saved successfully: {name}"
    except Exception as exc:
        return f"File saving error: {exc}"


TOOLS = [calculator, text_analyzer, save_note]


# --------------------------------------------------
# LangGraph
# --------------------------------------------------

def build_graph():
    llm = get_llm()
    tool_llm = llm.bind_tools(TOOLS)

    def planner_node(state: AgentState):
        return {"plan": generate_plan(state["user_request"])}

    def executor_node(state: AgentState):
        messages = state.get("messages", [])

        if not messages:
            messages = [
                SystemMessage(content="""
You are TaskPilot Executor.
Carry out the user's request using the available tools when needed.
Available tools: calculator, text_analyzer, save_note.
Use calculator for arithmetic, text_analyzer for text statistics,
and save_note when the user asks to save text to a file.
Do not claim a tool succeeded unless its actual tool output confirms it.
After tools finish, provide a concise final response.
"""),
                HumanMessage(content=(
                    f"User request:\n{state['user_request']}\n\n"
                    f"Task plan:\n{state['plan'].model_dump_json(indent=2)}\n\n"
                    "Execute the request."
                ))
            ]

        response = tool_llm.invoke(messages)

        # On the first call, add the system and human messages as well.
        if not state.get("messages"):
            return {"messages": messages + [response]}

        return {"messages": [response]}

    def reviewer_node(state: AgentState):
        transcript = []

        for message in state.get("messages", []):
            if isinstance(message, AIMessage):
                if message.tool_calls:
                    for call in message.tool_calls:
                        transcript.append(
                            f"Tool requested: {call['name']}; arguments: {call['args']}"
                        )
                elif message.content:
                    transcript.append(f"Agent response: {message.content}")

            elif isinstance(message, ToolMessage):
                transcript.append(
                    f"Tool result ({message.name}): {message.content}"
                )

        review_response = llm.invoke([
            SystemMessage(content="""
You are TaskPilot Reviewer.
Summarize what was actually completed.
Report actual tool outputs, incomplete work, and limitations.
Never invent results or claim that an action happened without evidence.
"""),
            HumanMessage(content=(
                f"User request:\n{state['user_request']}\n\n"
                f"Plan:\n{state['plan'].model_dump_json(indent=2)}\n\n"
                f"Execution transcript:\n{chr(10).join(transcript)}"
            ))
        ])

        return {"review": str(review_response.content)}

    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")

    graph.add_conditional_edges(
        "executor",
        tools_condition,
        {
            "tools": "tools",
            END: "reviewer"
        }
    )

    graph.add_edge("tools", "executor")
    graph.add_edge("reviewer", END)

    return graph.compile()


def execute_agent(user_request: str):
    agent = build_graph()

    result = agent.invoke(
        {
            "user_request": user_request,
            "messages": []
        },
        config={"recursion_limit": 15}
    )

    tool_outputs = [
        {
            "tool": message.name,
            "output": str(message.content)
        }
        for message in result.get("messages", [])
        if isinstance(message, ToolMessage)
    ]

    executed_tools = list(dict.fromkeys(
        item["tool"] for item in tool_outputs
    ))

    return {
        "plan": result["plan"].model_dump(),
        "executed_tools": executed_tools,
        "tool_outputs": tool_outputs,
        "review": result.get("review", "")
    }


# --------------------------------------------------
# Flask routes
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    user_request = ""

    if request.method == "POST":
        user_request = request.form.get("request", "").strip()
        action = request.form.get("action", "run")

        if len(user_request) < 3:
            error = "Please enter a task containing at least 3 characters."
        else:
            try:
                if action == "plan":
                    plan = generate_plan(user_request)
                    result = {
                        "plan": plan.model_dump(),
                        "executed_tools": [],
                        "tool_outputs": [],
                        "review": ""
                    }
                elif action == "run":
                    result = execute_agent(user_request)
                else:
                    error = "Invalid form action."
            except Exception as exc:
                app.logger.exception("TaskPilot request failed")
                error = str(exc)

    return render_template(
        "index.html",
        result=result,
        error=error,
        user_request=user_request
    )


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "success",
        "message": "TaskPilot is running"
    })


@app.route("/api/plan", methods=["GET", "POST"])
def create_plan():
    if request.method == "GET":
        return jsonify({
            "status": "info",
            "message": "This endpoint requires an HTTP POST request with a JSON body.",
            "usage": {
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": {"request": "Calculate total cost of 25 products, each costing 500 rupees."}
            },
            "web_ui": "Open http://127.0.0.1:5000 in your browser to use the graphical interface."
        }), 200

    try:
        data = request.get_json(silent=True) or {}
        user_request = str(data.get("request") or "").strip()

        if len(user_request) < 3:
            return jsonify({"error": "Enter a valid task containing at least 3 characters."}), 400

        plan = generate_plan(user_request)

        return jsonify({
            "status": "success",
            "plan": plan.model_dump()
        })
    except Exception as exc:
        app.logger.exception("Plan generation failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/run", methods=["GET", "POST"])
def run_agent():
    if request.method == "GET":
        return jsonify({
            "status": "info",
            "message": "This endpoint requires an HTTP POST request with a JSON body.",
            "usage": {
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": {"request": "Calculate total cost of 25 products, each costing 500 rupees."}
            },
            "web_ui": "Open http://127.0.0.1:5000 in your browser to use the graphical interface."
        }), 200

    try:
        data = request.get_json(silent=True) or {}
        user_request = str(data.get("request") or "").strip()

        if len(user_request) < 3:
            return jsonify({"error": "Enter a valid task containing at least 3 characters."}), 400

        return jsonify({
            "status": "success",
            **execute_agent(user_request)
        })
    except Exception as exc:
        app.logger.exception("Agent execution failed")
        return jsonify({"error": str(exc)}), 500


@app.errorhandler(405)
def method_not_allowed(e):
    if request.path.startswith("/api/"):
        return jsonify({
            "status": "error",
            "error": "Method Not Allowed. Please verify the HTTP method (GET vs POST).",
            "requested_url": request.url,
            "web_ui": "http://127.0.0.1:5000"
        }), 405
    return render_template(
        "index.html",
        result=None,
        error="Method Not Allowed for this URL. Please use the form on the home page (http://127.0.0.1:5000).",
        user_request=""
    ), 405


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":
    print("\nTaskPilot registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"{rule.rule:20} {sorted(rule.methods)}")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
