
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

app = Flask(__name__)


# ==========================================
# 2. PYDANTIC MODELS
# ==========================================

class TaskStep(BaseModel):

    step_number: int = Field(
        ge=1,
        description="Sequential step number"
    )

    title: str = Field(
        description="Short title of the task"
    )

    description: str = Field(
        description="Detailed explanation of the task"
    )

    estimated_minutes: int = Field(
        ge=1,
        description="Estimated time in minutes"
    )

    expected_output: str = Field(
        description="Expected result after completing this step"
    )


class TaskPlan(BaseModel):

    goal: str = Field(
        description="Main goal of the user"
    )

    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions made while creating the plan"
    )

    steps: list[TaskStep] = Field(
        min_length=1,
        description="Ordered list of task steps"
    )

    total_estimated_minutes: int = Field(
        ge=1,
        description="Total estimated time in minutes"
    )

    success_criteria: list[str] = Field(
        default_factory=list,
        description="Criteria for successful completion"
    )


# ==========================================
# 3. INITIALIZE GROQ LLM
# ==========================================

def get_llm():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing from .env file"
        )

    model_name = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b"
    )

    llm = ChatGroq(
        model=model_name,
        temperature=0,
        api_key=api_key
    )

    return llm


# ==========================================
# 4. HOME ROUTE
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return render_template("index.html")


# ==========================================
# 5. HEALTH CHECK
# ==========================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "success",
        "message": "TaskPilot API is running"
    })


# ==========================================
# 6. GENERATE TASK PLAN
# ==========================================

@app.route("/api/plan", methods=["POST"])
def create_plan():

    try:

        # Get request data

        data = request.get_json(silent=True) or {}

        user_request = data.get("request", "")

        user_request = user_request.strip()

        # Validate user input

        if len(user_request) < 3:

            return jsonify({
                "status": "error",
                "error": "Please enter a valid task."
            }), 400

        # Create prompt

        prompt = ChatPromptTemplate.from_messages([

            (
                "system",
                """
                You are TaskPilot, an intelligent AI task
                planning assistant.

                Your job is to convert user requests into
                practical and realistic task plans.

                STRICT INSTRUCTIONS:

                1. Understand the user's goal.
                2. Break the goal into actionable steps.
                3. Arrange steps in the correct order.
                4. Provide realistic time estimates in minutes.
                5. Every step must have an expected output.
                6. Include assumptions when necessary.
                7. Calculate total estimated time correctly.
                8. Define clear success criteria.
                9. Never claim that you executed a task.
                10. Return data according to the supplied schema.

                IMPORTANT FIELD REQUIREMENTS:

                Every step MUST contain:

                - step_number
                - title
                - description
                - estimated_minutes
                - expected_output

                Do not use alternative field names such as
                step, estimated_time_hours or time.

                The total_estimated_minutes must be the sum
                of all step estimated_minutes.

                Generate a practical and detailed plan.
                """
            ),

            (
                "human",
                """
                Create a detailed task plan for this request:

                {user_request}
                """
            )

        ])

        # Initialize LLM

        llm = get_llm()

        # Configure structured output

        structured_llm = llm.with_structured_output(
            TaskPlan,
            method="function_calling"
        )

        # Create LangChain pipeline

        chain = prompt | structured_llm

        # Generate plan

        plan = chain.invoke({
            "user_request": user_request
        })

        # Validate result

        if not isinstance(plan, TaskPlan):

            plan = TaskPlan.model_validate(plan)

        # Convert Pydantic object to dictionary

        result = plan.model_dump()

        # Return response

        return jsonify({
            "status": "success",
            "plan": result
        }), 200

    except Exception as exc:

        app.logger.exception(
            "TaskPilot plan generation failed"
        )

        return jsonify({
            "status": "error",
            "error": str(exc)
        }), 500


# ==========================================
# 7. RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
