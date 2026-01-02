# shema.py
from __future__ import annotations

from typing import Annotated, List, Optional, Literal, Any, Dict
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrueFalseAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    IsTrueOrFlase: bool = Field(
        ...,
        description="Indicates whether the statement in the question is true or false.",
    )


class MultiChoiceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Text: str = Field(..., min_length=1, description="The visible text of the answer option.")
    IsCorrect: bool = Field(..., description="Marks whether this option is a correct answer.")


class AnswerSchema(BaseModel):
    """
    Robust schema:
    - Accepts minor model mistakes (like MultiChoiceAnswer: [] on true/false) and normalizes them.
    - Still enforces the *required* parts (TF must have TrueFalseAnswers; MC must have options).
    """
    model_config = ConfigDict(extra="forbid")

    AnswerType: Literal[0, 1] = Field(
        ...,
        description="Defines the answer format. 0 = True/False, 1 = Multiple-choice.",
    )

    TrueFalseAnswers: Annotated[
        Optional[TrueFalseAnswer],
        Field(
            description=(
                "Used only when AnswerType is 0 (True/False). "
                "Contains the correct boolean answer."
            )
        ),
    ] = None

    MultiChoiceAnswer: Annotated[
        Optional[List[MultiChoiceItem]],
        Field(
            description=(
                "Used only when AnswerType is 1 (Multiple Choice). "
                "Contains all answer options and marks correct ones."
            )
        ),
    ] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, data: Any) -> Any:
        """
        Make it tolerant to common LLM glitches:
        - If AnswerType=0 and MultiChoiceAnswer is present (even []), drop it.
        - If AnswerType=1 and TrueFalseAnswers is present, drop it.
        """
        if not isinstance(data, dict):
            return data

        # Copy defensively (avoid mutating caller dict)
        d: Dict[str, Any] = dict(data)
        at = d.get("AnswerType")

        if at == 0:
            # True/False: MultiChoiceAnswer must not exist (normalize away)
            d.pop("MultiChoiceAnswer", None)
        elif at == 1:
            # Multiple-choice: TrueFalseAnswers must not exist (normalize away)
            d.pop("TrueFalseAnswers", None)

        return d

    @model_validator(mode="after")
    def _validate_shape(self) -> "AnswerSchema":
        if self.AnswerType == 0:
            if self.TrueFalseAnswers is None:
                raise ValueError("TrueFalseAnswers must be provided when AnswerType=0.")
            # Ensure clean shape
            self.MultiChoiceAnswer = None

        else:  # AnswerType == 1
            if not self.MultiChoiceAnswer:
                raise ValueError("MultiChoiceAnswer must be provided when AnswerType=1.")

            # Must have at least one correct option
            if not any(opt.IsCorrect for opt in self.MultiChoiceAnswer):
                raise ValueError("MultiChoiceAnswer must contain at least one IsCorrect=true option.")

            # Prevent duplicate option text (common LLM failure)
            seen = set()
            for opt in self.MultiChoiceAnswer:
                t = opt.Text.strip().lower()
                if t in seen:
                    raise ValueError("MultiChoiceAnswer contains duplicate Text options.")
                seen.add(t)

            # Ensure clean shape
            self.TrueFalseAnswers = None

        return self


class QuestionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Question: str = Field(..., min_length=3, description="The question text shown to the user.")
    WaitTimeInSec: int = Field(..., ge=1, le=60, description="Time in seconds the user has to answer the question.")
    Answer: AnswerSchema = Field(..., description="Defines the correct answer and its structure.")


class Quiz(BaseModel):
    """
    IMPORTANT:
    OpenAI Structured Outputs requires the top-level schema to be type 'object',
    so we wrap the array inside this object.
    """
    model_config = ConfigDict(extra="forbid")

    Questions: List[QuestionItem] = Field(
        ...,
        description="The quiz questions as a list. When saving, you can dump only this list to get a pure JSON array.",
    )
