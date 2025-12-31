# Quiz Generator

This project generates structured quiz questions using an LLM and validates the output with **Pydantic v2**.
The final result is saved as a pure JSON array that matches the required quiz schema.

---

## Features

- Generates quiz questions via an LLM
- Strict schema validation with Pydantic v2
- Supports:
  - True / False questions
  - Multiple-choice questions
- Ensures correct structure based on `AnswerType`
- Outputs a clean JSON array (`quiz.json`)
- Compatible with OpenAI Structured Outputs (`response_format`)

---

## Project Structure

```
quizGenerator/
│
├─ generator.py      # Main script that calls the LLM and saves quiz.json
├─ shema.py          # Pydantic schema definitions (intentionally named 'shema')
├─ quiz.json         # Generated output (created at runtime)
├─ README.md         # This file
└─ .venv/            # Virtual environment (not committed)
```

---

## Requirements

- Python **3.12+**
- `uv`
- An OpenAI-compatible API endpoint (local or cloud)

Python packages:
- pydantic >= 2.12
- openai
- agents
- python-dotenv
- ddgs

---

## Setup

1. Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate    # Windows
```

2. Install dependencies:

```bash
uv pip install pydantic openai agents python-dotenv ddgs
```

3. Create a `.env` file (if needed):

```env
OPENAI_API_KEY=your_api_key_here
```

---

## Usage

Run the generator:

```bash
uv run generator.py
```

This will:
1. Ask the LLM to generate a quiz
2. Validate the response against the schema
3. Save the result to `quiz.json`

---

## Output Format

The generated `quiz.json` is a **pure JSON array**, for example:

```json
[
  {
    "Question": "Magyarország államformája köztársaság.",
    "WaitTimeInSec": 10,
    "Answer": {
      "AnswerType": 0,
      "TrueFalseAnswers": {
        "IsTrueOrFlase": true
      }
    }
  }
]
```

### Answer Types

- `AnswerType = 0`
  - Uses `TrueFalseAnswers`
- `AnswerType = 1`
  - Uses `MultiChoiceAnswer`

The schema enforces that **only the correct answer block is present**.

---

## Important Notes

- The schema uses a wrapper object internally (`Quiz`) because OpenAI Structured Outputs require a top-level JSON object.
- When saving, only the `Questions` array is written to `quiz.json` to match the required format.
- Field names are intentionally kept identical to the target JSON (including `IsTrueOrFlase`).

---

## Troubleshooting

### Error: schema must be type "object"
This is expected behavior from OpenAI Structured Outputs.
The fix is already implemented by wrapping the array inside the `Quiz` model.

### Pydantic schema errors
Make sure:
- You are using **Pydantic v2**
- Optional fields do NOT use `Field(None, ...)`

---

## License

This project is for personal and educational use.
