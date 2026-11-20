TASKPILOT DAY 3 - COMPLETE PROJECT

1. Extract this folder.
2. Open the extracted folder in VS Code.
3. Create and activate a virtual environment (Windows PowerShell):

   python -m venv .venv
   .venv\Scripts\Activate.ps1

4. Install dependencies:

   pip install -r requirements.txt

5. Rename .env.example to .env and add your Groq API key.
   Keep the key private. Do not upload .env to GitHub.

6. Run:

   python app.py

7. Open:

   http://127.0.0.1:5000

WORKFLOW
- User enters a task.
- LangGraph creates a plan.
- Graph pauses at the human-approval interrupt.
- User approves or rejects in the browser.
- Only after approval does the executor run.
- Reviewer creates a structured report.
- Frontend renders the report as HTML tables.

IMPORTANT
- app.py and templates/index.html are a matching pair.
- Run this app.py, not an older day3/app.py from another folder.
- MemorySaver stores pending graph state in memory. Restarting Flask clears pending approvals.
- The app uses Groq at runtime; the source files are syntax-checked, but live API behavior depends on your API key, model access and installed package versions.
