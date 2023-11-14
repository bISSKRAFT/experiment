from src.prompts.few_shot import FewShotTemplate
import pytest

from src.prompts.prompt import PromptTemplate


def test_few_shot_value_error() -> None:
    """Test the formatting of a prompt."""
    prefix = PromptTemplate(
        template="This prefix has {prefix}",
        input_variables=["prefix"],
    )
    suffix = PromptTemplate(
        template="This suffix has {suffix}",
        input_variables=["suffix"],
    )
    example_prompt = PromptTemplate(
        template="{variable1}: {variable2}",
        input_variables=["variable1", "variable2"],
    )
    examples = [
        {"variable1": "value1", "variable2": "value2"},
        {"variable1": "value3", "variable2": "value4"},
    ]
    with pytest.raises(ValueError):
        FewShotTemplate(
            suffix=suffix,
            prefix=prefix,
            input_variables=[],
            examples=examples,
            examples_prompt=example_prompt,
            example_delimiter="\n",
            validation=True,
        )


def test_few_shot_format() -> None:
    prefix = PromptTemplate(
        template="This prefix has {prefix}",
        input_variables=["prefix"],
    )
    suffix = PromptTemplate(
        template="This suffix has {suffix}",
        input_variables=["suffix"],
    )
    example_prompt = PromptTemplate(
        template="{variable1}: {variable2}",
        input_variables=["variable1", "variable2"],
    )
    examples = [
        {"variable1": "value1", "variable2": "value2"},
        {"variable1": "value3", "variable2": "value4"},
    ]
    prompt = FewShotTemplate(
        suffix=suffix,
        prefix=prefix,
        input_variables=["prefix", "suffix"],
        examples=examples,
        examples_prompt=example_prompt,
        examples_delimiter="\n",
        validation=True,
    )
    prompt = prompt.format(prefix="prefix", suffix="suffix")
    expected_output = (
        "This prefix has prefix\n"
        "value1: value2\n"
        "value3: value4\n"
        "This suffix has suffix"
    )
    assert prompt == expected_output


def test_raise_error_if_stop_in_variable() -> None:
    with pytest.raises(ValueError):
        FewShotTemplate(
            suffix=PromptTemplate(
                template="This is a {stop} test",
                input_variables=["stop"],
            ),
            prefix=None,
            input_variables=["stop"],
            examples=[],
            examples_prompt=PromptTemplate(
                template="This is a {stop} test",
                input_variables=["stop"],
            ),
            examples_delimiter="\n",
            validation=True,
        )


def test_raise_error_no_examples() -> None:
    template = FewShotTemplate(
        suffix=PromptTemplate(
            template="This is a {placeholder} test",
            input_variables=["placeholder"],
        ),
        prefix=None,
        input_variables=["placeholder"],
        examples=None,
        examples_prompt=PromptTemplate(
            template="This is a {example_plh} example",
            input_variables=["example_plh"],
        ),
        examples_delimiter="\n",
        validation=True,
    )
    with pytest.raises(ValueError):
        template.format(placeholder="validation")
