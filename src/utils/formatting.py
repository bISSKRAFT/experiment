from string import Formatter
from typing import Sequence, Mapping, Any, List


class MyFormatter(Formatter):
    def vformat(
            self, format_string: str, args: Sequence, kwargs: Mapping[str, Any]
    ) -> str:
        """Check if args are provided"""
        if len(args):
            raise ValueError(
                "args not supported, "
                "use kwargs instead."
            )
        return super().vformat(format_string, args, kwargs)

    def validate_input_variables(
            self, format_string: str, input_variables: List[str]
    ) -> None:
        dummy_inputs = {input_variable: "foo" for input_variable in input_variables}
        super().format(format_string, **dummy_inputs)


formatter = MyFormatter()
