# Smart-TaskPilot: An AI-Powered Task Research and Action Agent

Smart-TaskPilot is an agentic AI assistant built with **Flask**, **LangChain**, **LangGraph**, and **Groq**. It transforms natural language task requests into structured, executable plans, coordinates tool usage, supports human-in-the-loop review, and synthesizes final action reports.

---

The repository is structured across three iterative development phases:

### [Day 1: Structured Planning & Schema Enforcement](file:///Day1)
- **Goal**: Convert open-ended user requests into validated, structured task plans using LangChain and Pydantic (`TaskPlan` schema).
- **Key Concepts**: System prompt engineering, structured output with Groq, schema validation (`total_estimated_minutes`, steps, assumptions, success criteria), resilient fallback handling.
- **Run**:
  ```bash
  cd Day1
  pip install -r requirements.txt
  cp .env.example .env   # configure your GROQ_API_KEY
  python app.py
  ```

### [Day 2: LangGraph Agent & Tool Execution](file:///Day2)
- **Goal**: Implement a multi-node agent workflow using LangGraph to plan, execute, and review tasks using built-in tools.
- **Key Concepts**: StateGraph (`AgentState`), tool binding (`calculator`, `text_analyzer`, `save_note`), conditional tool routing (`tools_condition`), execution reviews.
- **Run**:
  ```bash
  cd Day2
  pip install -r requirements.txt
  cp .env.example .env   # configure your GROQ_API_KEY
  python app.py
  ```

### [Day 3: Human-in-the-Loop & State Checkpointing](file:///Day3)
- **Goal**: Add approval checkpoints and state persistence using LangGraph's `MemorySaver` and `interrupt`.
- **Key Concepts**: Human-in-the-loop approval workflows, interrupt states (`__interrupt__`), thread-based execution resumption (`Command(resume=...)`), approval/rejection branches.
- **Run**:
  ```bash
  cd Day3
  pip install -r requirements.txt
  cp .env.example .env   # configure your GROQ_API_KEY
  python app.py
  ```

---

## 🛠️ Setup & Configuration

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MohitPayasi/Smart-TaskPilot-An-AI-Powered-Task-Research-and-Action-Agent.git
   cd Smart-TaskPilot-An-AI-Powered-Task-Research-and-Action-Agent
   ```

2. **Environment Configuration:**
   Each day folder contains an `.env.example` file. Create a `.env` in the respective directory and add your Groq API credentials:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=openai/gpt-oss-120b
   ```

3. **Install Dependencies & Start Application:**
   Navigate to any day's directory, install requirements, and run `python app.py`. Visit `http://127.0.0.1:5000` in your browser.
