from __future__ import annotations

from typing import Annotated, List, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrueFalseAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    IsTrueOrFlase: bool = Field(
        ...,
        description="Indicates whether the statement in the question is true or false.",
    )


class MultiChoiceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Text: str = Field(
        ...,
        description="The visible text of the answer option.",
    )
    IsCorrect: bool = Field(
        ...,
        description="Marks whether this option is a correct answer.",
    )


class AnswerSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    AnswerType: Literal[0, 1] = Field(
        ...,
        description="Defines the answer format. 0 = True/False, 1 = Multiple-choice.",
    )

    # NOTE: default None is outside Field() to avoid the NoneType schema crash in Pydantic v2
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

    @model_validator(mode="after")
    def _validate_shape(self) -> "AnswerSchema":
        if self.AnswerType == 0:
            if self.TrueFalseAnswers is None:
                raise ValueError("TrueFalseAnswers must be provided when AnswerType=0.")
            if self.MultiChoiceAnswer is not None:
                raise ValueError("MultiChoiceAnswer must be omitted/null when AnswerType=0.")
        else:  # AnswerType == 1
            if not self.MultiChoiceAnswer:
                raise ValueError("MultiChoiceAnswer must be provided when AnswerType=1.")
            if self.TrueFalseAnswers is not None:
                raise ValueError("TrueFalseAnswers must be omitted/null when AnswerType=1.")
        return self


class QuestionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Question: str = Field(
        ...,
        description="The question text shown to the user.",
    )

    WaitTimeInSec: int = Field(
        ...,
        ge=1,
        description="Time in seconds the user has to answer the question.",
    )

    Answer: AnswerSchema = Field(
        ...,
        description="Defines the correct answer and its structure.",
    )


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
