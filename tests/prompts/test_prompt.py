from src.prompts.prompt import PromptTemplate
import pytest


def test_prompt_creation() -> None:
    """Test the creation of a prompt."""
    template = "this is a {placeholder} test"
    input_variables = ["placeholder"]
    prompt = PromptTemplate(input_variables=input_variables, template=template)
    assert prompt.template == template
    assert prompt.input_variables == input_variables


def test_format() -> None:
    """Test the formatting of a prompt."""
    template = "this is a {placeholder} test"
    input_variables = ["placeholder"]
    prompt = PromptTemplate(input_variables=input_variables, template=template)
    assert prompt.format(placeholder="formatted") == "this is a formatted test"


def test_prompt_missing_input_variables() -> None:
    """Test the formatting of a prompt."""
    template = "this is a {placeholder} test"
    input_variables = []
    with pytest.raises(ValueError):
        PromptTemplate(input_variables=input_variables, template=template, validation=True)


def test_raise_error_if_stop_in_variable() -> None:
    with pytest.raises(ValueError):
        PromptTemplate(
            template="This is a {stop} test",
            input_variables=["stop"],
        )
