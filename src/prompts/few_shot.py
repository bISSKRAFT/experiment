from typing import List, Optional, Dict, Any

from pydantic.v1 import root_validator

from src.prompts.prompt import BaseTemplate, PromptTemplate
from src.utils.formatting import formatter


class FewShotTemplate(BaseTemplate):
    """Few-shot template."""
    input_variables: List[str]
    validation: bool = False
    prefix: Optional[PromptTemplate] = None
    examples_prompt: PromptTemplate
    examples_delimiter: str = "\n\n"
    examples: Optional[List[dict]] = None
    suffix: PromptTemplate

    @root_validator()
    def template_validation(cls, values: Dict) -> Dict:
        if values["validation"]:
            input_variables = values["input_variables"]
            expected_variables = set(values["suffix"].input_variables)
            if values["prefix"]:
                expected_variables |= set(values["prefix"].input_variables)
            mismatch = expected_variables.difference(input_variables)
            if mismatch:
                raise ValueError(
                    "Input variables do not match. "
                    f"Expected: {expected_variables}, "
                    f"got: {input_variables}."
                )
        return values

    def _fetch_examples(self) -> List[dict]:
        if self.examples is not None:
            return self.examples
        else:
            raise ValueError("No examples provided.")

    def format(self, **kwargs: Any) -> str:
        examples = self._fetch_examples()
        example_prompts = [
            self.examples_prompt.format(**example)
            for example in examples
        ]
        if self.prefix is None:
            prefix_prompt = ""
        else:
            prefix_kwargs = {
                k: v for k, v in kwargs.items()
                if k in self.prefix.input_variables
            }
            for k in prefix_kwargs.keys():
                kwargs.pop(k)
            prefix_prompt = self.prefix.format(**prefix_kwargs)

        suffix_kwargs = {
            k: v for k, v in kwargs.items()
            if k in self.suffix.input_variables
        }
        for k in suffix_kwargs.keys():
            kwargs.pop(k)
        suffix_prompt = self.suffix.format(**suffix_kwargs)

        pieces = [prefix_prompt, *example_prompts, suffix_prompt]
        full_template = self.examples_delimiter.join(pieces)
        return formatter.format(full_template, **kwargs)
