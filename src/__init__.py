# from typing import Any
#
#
# def __getattr__(name: str) -> Any:
#     if name == "InferenceLLM":
#         from src.models.llms import InferenceLLM
#         return InferenceLLM
#     elif name == "BaseLLM":
#         from src.models.llms import BaseLLM
#         return BaseLLM
