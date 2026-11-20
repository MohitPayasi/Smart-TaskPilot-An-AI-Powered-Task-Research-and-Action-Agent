
import os

from dotenv import load_dotenv

from flask import (
    Flask,
    jsonify,
    render_template,
    request
)

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
        description="Step number in sequence"
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
# 3. GROQ LLM CONFIGURATION
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
# 5. HEALTH CHECK ROUTE
# ==========================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({

        "status": "success",

        "message": "TaskPilot API is running"

    })


# ==========================================
# 6. TASK PLAN GENERATION
# ==========================================

@app.route("/api/plan", methods=["POST"])
def create_plan():

    try:

        # Get JSON data from frontend

        data = request.get_json(silent=True) or {}

        user_request = data.get("request", "")

        user_request = user_request.strip()


        # Validate user input

        if len(user_request) < 3:

            return jsonify({

                "status": "error",

                "error": "Please enter a valid task."

            }), 400


        # Create Prompt Template

        prompt = ChatPromptTemplate.from_messages([

            (
                "system",
                """
                You are TaskPilot,
                an intelligent AI task planning assistant.

                Your job is to convert user requests
                into practical and realistic task plans.

                Follow these instructions:

                1. Understand the user's goal.

                2. Break the goal into actionable steps.

                3. Arrange steps in the correct order.

                4. Provide realistic time estimates.

                5. Every step must have an expected output.

                6. Include assumptions when necessary.

                7. Calculate total estimated time correctly.

                8. Define clear success criteria.

                9. Do not claim that you have executed tasks.

                10. Return structured JSON data only.

                Do not return normal conversational text.
                """
            ),

            (
                "human",
                """
                Create a detailed task plan
                for the following request:

                {user_request}
                """
            )

        ])


        # Configure Structured Output

        llm = get_llm()

        structured_llm = llm.with_structured_output(

            TaskPlan,

            method="json_mode"

        )


        # Create LangChain Pipeline

        chain = prompt | structured_llm


        # Invoke LLM

        plan = chain.invoke({

            "user_request": user_request

        })


        # Convert Pydantic Model into Dictionary

        result = plan.model_dump()


        # Return Successful Response

        return jsonify({

            "status": "success",

            "plan": result

        }), 200


    except Exception as exc:

        # Log complete error in terminal

        app.logger.exception(
            "TaskPilot plan generation failed"
        )


        # Return error response

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
