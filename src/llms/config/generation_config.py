from transformers import GenerationConfig


class GenerationConfigMixin:

    def __init__(self, **kwargs):
        self.generation_config = kwargs

    def to_hf_generation_config(self):
        return GenerationConfig(
            **self.generation_config
        )