from typing import Any, Dict

from pydantic.v1 import root_validator

from src.models.prompts.base import BaseTemplate, check_valid_template
from src.utils.formatting import formatter


class PromptTemplate(BaseTemplate):
    """Prompt template with input variables."""

    validation = False
    """Whether to validate the template and input variables."""

    template: str
    """Template string."""

    def format(self, **kwargs: Any) -> str:
        return formatter.format(self.template, **kwargs)

    @root_validator()
    def template_validation(cls, values: Dict) -> Dict:
        if values["validation"]:
            check_valid_template(values["template"], values["input_variables"])
        return values
