import asyncio
import json
from typing import Any, Optional
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, trace
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ddgs import DDGS

from shema import Quiz

load_dotenv(override=True)

search = DDGS()

@function_tool
def search_on_duckduckgo(query: str, region: str, timelimit: Optional[str] = None, max_results: Optional[int] = 5) -> list[dict[str, Any]]:
    """
    Perform a text search on duckduckgo.com

    Args:
            query: The search query.
            region: The region to use for the search (e.g., us-en, uk-en, ru-ru, etc.).
            timelimit: The timelimit for the search (e.g., d, w, m, y) or custom date range.
            max_results: The maximum number of results to return. Defaults to 10.
        Returns:
            A list of dictionaries containing the search results.
    """
    return search.text(query=query, max_results=max_results, region=region, safesearch="off", timelimit=timelimit)

@function_tool
def search_on_duckduckgo_news(query: str, region: str = "us-en", timelimit: Optional[str] = None, max_results: Optional[int] = 5) -> list[dict[str, Any]]:
    """
    Perform a news search on duckduckgo.com

    Args:
            query: The search query.
            region: The region to use for the search (e.g., us-en, uk-en, ru-ru, etc.).
            timelimit: The timelimit for the search (e.g., d, w, m, y) or custom date range.
            max_results: The maximum number of results to return. Defaults to 10.
        Returns:
            A list of dictionaries containing the search results.
    """
    return search.news(query=query, max_results=max_results, region=region, safesearch="off", timelimit=timelimit)

@function_tool
def search_on_duckduckgo_books(query: str, region: str = "us-en", timelimit: Optional[str] = None, max_results: Optional[int] = 5) -> list[dict[str, Any]]:
    """
    Perform a books on duckduckgo.com

    Args:
            query: The search query.
            region: The region to use for the search (e.g., us-en, uk-en, ru-ru, etc.).
            timelimit: The timelimit for the search (e.g., d, w, m, y) or custom date range.
            max_results: The maximum number of results to return. Defaults to 10.
        Returns:
            A list of dictionaries containing the search results.
    """
    return search.books(query=query, max_results=max_results, region=region, safesearch="off", timelimit=timelimit)


async def main(model = None):
    INSTRUCTION = """
    You are a quiz generator assistant that creates humorous, engaging quizzes based on a strict JSON schema provided by the user. You accept a theme and a language from the user and generate questions relevant to the theme, written in the requested language. When information is available, you base facts on reliable public internet sources and include a short list of source links after the quiz. You always output a valid, copy-ready JSON array inside a proper code block that exactly matches the given structure, with correct field names and types. You vary question types between true/false and multiple choice, set reasonable wait times, and clearly mark correct answers. Keep humor light, friendly, and appropriate. If the user does not provide a theme or language, ask for them before generating the quiz. Prefer accuracy over jokes when there is a conflict, and never invent sources; if no sources are available, state that clearly.

    Example of the expected output format:
    ```
    [
    {
        "Question": "The sky is blue.",
        "WaitTimeInSec": 10,
        "Answer": {
        "AnswerType": 0,
        "TrueFalseAnswers": {
            "IsTrueOrFlase": true
        }
        }
    },
    {
        "Question": "Which one is a programming language?",
        "WaitTimeInSec": 12,
        "Answer": {
        "AnswerType": 1,
        "MultiChoiceAnswer": [
            { "Text": "Banana", "IsCorrect": false },
            { "Text": "C#", "IsCorrect": true },
            { "Text": "Table", "IsCorrect": false },
            { "Text": "Rust", "IsCorrect": true }
        ]
        }
    }
    ]
    ```
    Always include this formatting for any quiz you produce, and always add sources if available, or explicitly state if no sources were found.
    
    
    Output minimum 10 Question answers. Make sure true False Answers are not always true.
    """

    prompts = []

    with open("thema.json", "r", encoding="utf-8") as file:
        data = file.read()
        
        d = json.loads(data)
        prompts = d["thema"]
    
    if(model is None):
        model = OpenAIChatCompletionsModel( 
            model="functiongemma:270m",
            openai_client=AsyncOpenAI(base_url="http://localhost:11434/v1")
        )      
    
    agent = Agent(name="Quiz Generator", instructions=INSTRUCTION, tools=[search_on_duckduckgo, search_on_duckduckgo_books, search_on_duckduckgo_news], model=model, output_type=Quiz)
    res = None
    with trace("Quiz Generator"):
        i = 1
        for prompt in prompts:
            print("-"*12)
            print(f"User instruction: {prompt}.")
            res = await Runner.run(agent, input=prompt)
    
            quiz: Quiz = res.final_output
            
            # Dump ONLY the array to match your required output format:
            quiz_array = quiz.model_dump(mode="json", exclude_none=True)["Questions"]

            with open(f"{i}_quiz.json", "w", encoding="utf-8") as f:
                json.dump(quiz_array, f, ensure_ascii=False, indent=2)

            # Optional: print to console
            print(json.dumps(quiz_array, ensure_ascii=False, indent=2))
            i = i + 1


if __name__ == "__main__":
    mode = "gpt-5-nano"
    asyncio.run(main(model=mode))