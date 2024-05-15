#############################################################################################
#                                                                                           #
#       Inspired by LangChain https://github.com/langchain-ai/langchain                     #
#                                                                                           #
#                                                                                           #
#############################################################################################




from abc import ABC, abstractmethod
from typing import List, Any
from pydantic.v1 import BaseModel, root_validator

from src.utils.formatting import formatter


class BaseTemplate(BaseModel, ABC):
    input_variables: List[str]

    @abstractmethod
    def format(self, **kwargs: Any) -> str:
        pass

    @root_validator()
    def validate_variable_names(cls, values: dict) -> dict:
        if "stop" in values["input_variables"]:
            raise ValueError("stop is a reserved keyword")
        return values


def check_valid_template(
        template: str,
        input_variables: List[str]
) -> None:
    try:
        formatter.validate_input_variables(template, input_variables)
    except KeyError as e:
        raise ValueError(
            "Invalid template or input variables. ",
            "Check the template and input variables. " + str(e)
        )
