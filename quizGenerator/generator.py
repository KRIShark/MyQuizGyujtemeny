# generator.py
import asyncio
import json
import re
from typing import Any, Optional, List, Dict

from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, RunConfig, Runner, SQLiteSession, function_tool, trace
from agents.exceptions import ModelBehaviorError
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ddgs import DDGS
import urllib
import wikipediaapi

from shema import Quiz

load_dotenv(override=True)
search = DDGS()


@function_tool
def search_on_duckduckgo(
    query: str,
    region: str = "hu-hu",
    timelimit: Optional[str] = None,
    max_results: Optional[int] = 5,
) -> list[dict[str, Any]]:
    return search.text(
        query=query,
        max_results=max_results,
        region=region,
        safesearch="moderate",
        timelimit=timelimit,
    )


@function_tool
def search_on_duckduckgo_news(
    query: str,
    region: str = "hu-hu",
    timelimit: Optional[str] = None,
    max_results: Optional[int] = 5,
) -> list[dict[str, Any]]:
    return search.news(
        query=query,
        max_results=max_results,
        region=region,
        safesearch="moderate",
        timelimit=timelimit,
    )


@function_tool
def search_on_duckduckgo_books(
    query: str,
    region: str = "hu-hu",
    timelimit: Optional[str] = None,
    max_results: Optional[int] = 5,
) -> list[dict[str, Any]]:
    return search.books(
        query=query,
        max_results=max_results,
        region=region,
        safesearch="moderate",
        timelimit=timelimit,
    )
    
@function_tool
def wikipedia_search(
    query: str,
    lang: str = "hu",
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    REAL Wikipedia search using MediaWiki API.
    Returns article titles that actually exist.
    """
    import urllib.parse
    import urllib.request
    import json

    query = (query or "").strip()
    if not query:
        return []

    limit = max(1, min(limit, 10))

    url = (
        f"https://{lang}.wikipedia.org/w/api.php"
        "?action=query"
        "&list=search"
        "&format=json"
        "&utf8=1"
        f"&srlimit={limit}"
        f"&srsearch={urllib.parse.quote(query)}"
    )

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    results = []
    for item in data.get("query", {}).get("search", []):
        title = item.get("title")
        if not title:
            continue

        results.append({
            "title": title,
            "snippet": re.sub(r"<[^>]+>", "", item.get("snippet", "")),
            "url": f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
        })

    return results


@function_tool
def wikipedia_summary(
    title: str,
    lang: str = "hu",
    max_sentences: int = 3,
) -> dict[str, Any]:
    """
    Get a clean summary for a Wikipedia page.
    Always returns plain text if the page exists.
    """
    title = (title or "").strip()
    if not title:
        return {
            "title": "",
            "summary": "",
            "exists": False,
            "url": "",
        }

    wiki = _get_wiki(lang)
    page = wiki.page(title)

    if not page.exists():
        return {
            "title": title,
            "summary": "",
            "exists": False,
            "url": "",
        }

    summary = page.summary.strip()

    # Sentence limiting (safe heuristic)
    if summary and max_sentences > 0:
        parts = re.split(r"(?<=[.!?])\s+", summary)
        summary = " ".join(parts[:max_sentences])

    return {
        "title": page.title,
        "summary": summary,
        "exists": True,
        "url": page.fullurl,
    }


_WIKI_CACHE: dict[str, wikipediaapi.Wikipedia] = {}


def _get_wiki(lang: str) -> wikipediaapi.Wikipedia:
    if lang not in _WIKI_CACHE:
        _WIKI_CACHE[lang] = wikipediaapi.Wikipedia(
            language=lang,
            user_agent="QuizGenerator/1.0 (educational use)"
        )
    return _WIKI_CACHE[lang]


def _sanitize_quiz(quiz: Quiz) -> Quiz:
    # Extra safety net (schema already enforces this, but keep it anyway)
    for q in quiz.Questions:
        if q.Answer.AnswerType == 0:
            q.Answer.MultiChoiceAnswer = None
        else:
            q.Answer.TrueFalseAnswers = None
    return quiz


def _extract_json_object_from_error(msg: str) -> Optional[str]:
    # agents often embeds: 'Invalid JSON when parsing { ... } for TypeAdapter(Quiz)'
    m = re.search(r"parsing\s+(\{.*)\s+for\s+TypeAdapter", msg, flags=re.DOTALL)
    if not m:
        return None

    candidate = m.group(1).strip()
    last = candidate.rfind("}")
    if last == -1:
        return None
    return candidate[: last + 1]


def _build_system_prompt(min_questions: int) -> str:
    # Must match the structured output schema: top-level object with "Questions"
    return f"""
You are a quiz generator. You MUST output ONLY a JSON object that matches this schema exactly:

{{
  "Questions": [
    {{
      "Question": string,
      "WaitTimeInSec": integer,
      "Answer": {{
        "AnswerType": 0 or 1,
        "TrueFalseAnswers": {{ "IsTrueOrFlase": boolean }}   (ONLY if AnswerType = 0),
        "MultiChoiceAnswer": [{{"Text": string, "IsCorrect": boolean}}, ...] (ONLY if AnswerType = 1)
      }}
    }}
  ]
}}

You have tools available for research:
- wikipedia_summary(title, lang, max_sentences)
- search_on_duckduckgo / news / books

RESEARCH GUIDANCE:
- Use Wikipedia tools for encyclopedic facts (geography, history, science, etc.).
- Use DuckDuckGo if Wikipedia is missing details or for broader / current context.
- Never invent facts if unsure; prefer safer general-knowledge questions.

STRICT RULES (follow exactly):
- Output MUST be valid JSON, with double quotes everywhere.
- Output MUST be ONLY the JSON object (no markdown, no code blocks, no comments, no extra text).
- Produce at least {min_questions} questions.
- Language and style: follow the user instruction precisely.
- Mix question types: include both AnswerType=0 (true/false) and AnswerType=1 (multiple choice).
- If AnswerType=0:
  - Include TrueFalseAnswers
  - DO NOT include MultiChoiceAnswer at all (do not include it as [] or null)
- If AnswerType=1:
  - Include MultiChoiceAnswer with exactly 4 options
  - Mark exactly ONE option as IsCorrect=true
  - DO NOT include TrueFalseAnswers at all
- WaitTimeInSec: choose a reasonable value between 8 and 20.
- True/false questions: ensure some are true and some are false (not all true).
- Avoid duplicates and avoid ambiguous trick wording.
- Always use yout tools to search the Intenet for the resources to make sure your quizes are correct. If you dont find infromation try to use your knowlage as best as you can.
- You MUST use the web seach tools to at list try to find relevant knowlage for the quiz thema.

If you are unsure about a fact, ask a safer, general-knowledge question rather than inventing.
""".strip()


def _validate_content_rules(quiz: Quiz, min_questions: int) -> None:
    if len(quiz.Questions) < min_questions:
        raise ValueError(f"Too few questions: {len(quiz.Questions)} < {min_questions}")

    seen_q = set()
    has_tf = False
    has_mc = False

    for qi in quiz.Questions:
        qtxt = qi.Question.strip().lower()
        if qtxt in seen_q:
            raise ValueError("Duplicate questions detected.")
        seen_q.add(qtxt)

        if qi.Answer.AnswerType == 0:
            has_tf = True
            if qi.Answer.TrueFalseAnswers is None:
                raise ValueError("AnswerType=0 must have TrueFalseAnswers.")
            if qi.Answer.MultiChoiceAnswer is not None:
                raise ValueError("AnswerType=0 must not have MultiChoiceAnswer.")
        else:
            has_mc = True
            if not qi.Answer.MultiChoiceAnswer or len(qi.Answer.MultiChoiceAnswer) != 4:
                raise ValueError("AnswerType=1 must have exactly 4 multi choice options.")
            correct = sum(1 for opt in qi.Answer.MultiChoiceAnswer if opt.IsCorrect)
            if correct != 1:
                raise ValueError("AnswerType=1 must have exactly one correct option.")
            if qi.Answer.TrueFalseAnswers is not None:
                raise ValueError("AnswerType=1 must not have TrueFalseAnswers.")

    if not has_tf or not has_mc:
        raise ValueError("Quiz must include both true/false and multiple-choice questions.")


async def _run_with_retries(
    agent: Agent,
    user_prompt: str,
    min_questions: int,
    max_attempts: int = 4,
) -> Quiz:
    prompt = user_prompt

    session = SQLiteSession("QuizGenAgent")

    for attempt in range(1, max_attempts + 1):
        try:
            res = await Runner.run(agent, input=prompt, max_turns=50, session=session)
            quiz: Quiz = res.final_output
            quiz = _sanitize_quiz(quiz)
            _validate_content_rules(quiz, min_questions=min_questions)
            return quiz

        except ModelBehaviorError as e:
            raw = _extract_json_object_from_error(str(e))
            if raw:
                prompt = (
                    "You previously produced invalid JSON for the required schema.\n"
                    "Fix it and output ONLY the corrected JSON object.\n\n"
                    f"Original user request:\n{user_prompt}\n\n"
                    "Invalid JSON you produced:\n"
                    f"{raw}\n\n"
                    "Fix rules:\n"
                    "- Remove fields that are not allowed for the given AnswerType.\n"
                    "- Ensure at least the minimum number of questions.\n"
                    "- Ensure multiple-choice has 4 options and exactly one correct.\n"
                    "- Output ONLY JSON.\n"
                )
            else:
                prompt = (
                    f"{user_prompt}\n\n"
                    "IMPORTANT: Output ONLY the JSON object. "
                    "Do not include MultiChoiceAnswer when AnswerType=0, and do not include TrueFalseAnswers when AnswerType=1."
                )

            print(f"[Attempt {attempt}/{max_attempts}] ModelBehaviorError -> retrying with repair prompt.")

        except Exception as e:
            prompt = (
                f"{user_prompt}\n\n"
                "IMPORTANT: Follow the schema strictly and output ONLY JSON. "
                f"Previous error: {type(e).__name__}: {e}"
            )
            print(f"[Attempt {attempt}/{max_attempts}] {type(e).__name__} -> retrying.")

    #raise RuntimeError(f"Failed to generate a valid quiz after {max_attempts} attempts.")
    return None


def _sanitize_filename(name: str) -> str:
    """
    Windows-safe filename:
    - keep letters, digits, underscore, dash
    - replace everything else with underscore
    - collapse multiple underscores
    - avoid empty names
    """
    s = (name or "").strip()
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "quiz"


async def main(
    model_name: Optional[str] = None,
    prompts_path: str = "thema.json",
    min_questions: int = 10,
) -> None:
    system_prompt = _build_system_prompt(min_questions=min_questions)

    with open(prompts_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    entries = data.get("thema", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"No entries found in '{prompts_path}' under key 'thema'.")

    # Model setup
    base_client = AsyncOpenAI(base_url="http://localhost:11434/v1")
    if model_name is None:
        model = OpenAIChatCompletionsModel(model="gpt-oss:20b", openai_client=base_client)
    else:
        model = OpenAIChatCompletionsModel(model=model_name, openai_client=base_client)

    agent = Agent(
        name="Quiz Generator",
        instructions=system_prompt,
        tools=[search_on_duckduckgo, search_on_duckduckgo_books, search_on_duckduckgo_news, wikipedia_summary],
        model=model,
        output_type=Quiz,
        model_settings=ModelSettings(temperature=0.2, top_p=1.0)
    )

    # Track duplicates
    used_names: Dict[str, int] = {}

    with trace("Quiz Generator"):
        for entry in entries:
            if not isinstance(entry, dict):
                print("Skipping invalid entry (not an object).")
                continue

            raw_name = entry.get("name")
            instruction = entry.get("instruction")

            if not isinstance(raw_name, str) or not raw_name.strip():
                print("Skipping entry: missing/invalid 'name'.")
                continue
            if not isinstance(instruction, str) or not instruction.strip():
                print(f"Skipping entry '{raw_name}': missing/invalid 'instruction'.")
                continue

            safe_base = _sanitize_filename(raw_name)

            # Deduplicate filename if needed
            used_names[safe_base] = used_names.get(safe_base, 0) + 1
            if used_names[safe_base] > 1:
                safe_name = f"{safe_base}_{used_names[safe_base]}"
            else:
                safe_name = safe_base

            out_path = f"quiz/{safe_name}.json"

            print("-" * 12)
            print(f"Name: {raw_name}")
            print(f"Instruction: {instruction}")

            quiz = await _run_with_retries(
                agent,
                user_prompt=instruction,   # <— USE instruction as prompt
                min_questions=min_questions,
                max_attempts=4,
            )
            
            if quiz == None:
                print(f"Failed to generate {raw_name}.")
            else:
                # Dump ONLY the array to match your required output format:
                quiz_array = quiz.model_dump(mode="json", exclude_none=True)["Questions"]

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(quiz_array, f, ensure_ascii=False, indent=2)

                print(json.dumps(quiz_array, ensure_ascii=False, indent=2))
                print(f"Saved: {out_path}")


if __name__ == "__main__":
    # Optional CLI: uv run .\generator.py gpt-oss:20b
    import sys

    cli_model = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(model_name=cli_model, prompts_path="thema.json", min_questions=10))
