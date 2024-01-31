import warnings
import openai
import re
from openai.types.chat import ChatCompletion
from dotenv import load_dotenv
import os
from typing import Dict, List, Optional, Tuple
from src.models.quality.base import QualityScorerBase

ENV = load_dotenv("/proj/experiment/.env")
### G-EVAL INSTRUCTIONS ###


_COHERENCE = """
You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby "the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic."

Evaluation Steps:

1. Read the news article carefully and identify the main topic and key points.
2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.


Example:


Source Text:

{{Document}}

Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Coherence:
"""

_CONSISTENCY = """
You will be given a news article. You will then be given one summary written for this article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


Evaluation Criteria:

Consistency (1-5) - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts. 

Evaluation Steps:

1. Read the news article carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
3. Assign a score for consistency based on the Evaluation Criteria.


Example:


Source Text: 

{{Document}}

Summary: 

{{Summary}}


Evaluation Form (scores ONLY):

- Consistency:
"""

_FLUENCY = """
You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


Evaluation Criteria:

Fluency (1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.

- 1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
- 2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
- 3: Good. The summary has few or no errors and is easy to read and follow.


Example:

Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Fluency (1-3):
"""

_RELEVANCE = """
You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Relevance (1-5) - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information.

Evaluation Steps:

1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the article.
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.


Example:


Source Text:

{{Document}}

Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Relevance:
"""

### END G-EVAL INSTRUCTIONS ###




class GEvalQualityScorer(QualityScorerBase):
    """
    All Credit to https://github.com/nlpyang/geval

    Wrapper for using G-Eval as a quality scorer
    """
    _PROBA_ESTIMATE = float(1/20)

    def __init__(self,
                model_name: str = "gpt-3.5-turbo-1106"
    ) -> None:
        self.name = "GEvalQualityScorer"
        self.model_name = model_name
        if ENV is False:
            raise RuntimeError("No .env file found")

    def _create_client(self,
                       api_key: Optional[str] = os.getenv("USER_TOKEN"),
                       base_url: Optional[str] = os.getenv("SERVER_URL"),
    ) -> openai.OpenAI:
        """
        Creates a client for the OpenAI API

        Args:
            api_key (str): OpenAI API key
            base_url (str): OpenAI base URL
        """
        if api_key is None:
            raise ValueError("api_key must be provided")
        if base_url is None:
            raise ValueError("base_url must be provided")
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        return client

    def _check_for_digits(self, responses: List[str]) -> Tuple[List[str], int]:
        res_count = len(responses)
        for response in responses:
            if not any(char.isdigit() for char in response):
                warnings.warn(f"Response: --{response}-- does not contain digits")
                res_count -= 1
                responses.remove(response)

        return responses, res_count
    
    def _parse_output(self, output):
        matched = re.search("^ ?([\d\.]+)", output)
        if (matched):
            try:
                score = float(matched.group(1))
            except:
                score = 0
        else:
            score = 0
        return score

    def _calucate_score(self, responses: List[str], count: int) -> float:
        # create score like that: score = ∑p(si) *si
        prop = self._PROBA_ESTIMATE
        if count != 20:
            prop = float(1/count)
        print(responses)
        try:
            score = sum(
                [
                    self._parse_output(response) * prop
                    for response in responses
                ]
            )
        except ValueError as e:
            raise ValueError(f"Could not convert {responses} to float: {e}") from e
        return score

    def _make_reqeuest(self, prompt: str) -> Tuple[List[str], int]:
        client = self._create_client()

        try :
          _response: ChatCompletion = client.chat.completions.create(
              model=self.model_name,
              messages=[{"role": "system", "content": prompt}],
              temperature=2,
              max_tokens=5,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0,
              stop=None,
              n=20,
          )
        except openai.OpenAIError as e:
            raise RuntimeError(f"OpenAIError: {e}") from e

       # time.sleep(1)

        response_count = len(_response.choices)

        if response_count != 20:
            warnings.warn("Response does not contain 20 choices for probabilitiy estimation")
        res = [choice.message.content for choice in _response.choices if choice.message.content is not None]
        if len(res) != response_count:
            raise ValueError("All responses must contain a value")
        filtered_res, count = self._check_for_digits(res)
        if count == 0:
            raise ValueError("No responses contain digits")
        return filtered_res, count
        
    def _fill_prompt(self, prompt: str, source: str, summary: str, reference: Optional[str] = None) -> str:
        return prompt.replace('{{Document}}', source).replace('{{Summary}}', summary)
    
    def compute_score(self,
                      candidates: str,
                      source: str,
                      reference: Optional[str] = None,
    ) -> Dict:
        """
        Uses GPT-4 to score coherence, consistency, fluency and relevance of a summary
        
        Args:
            candidates (str): summary
            reference (str): reference text
            source (str): source text
        """
        dims = {
            "coherence": _COHERENCE,
            "consistency": _CONSISTENCY,
            "fluency": _FLUENCY,
            "relevance": _RELEVANCE,
        }
        result = {}
        for dim_name, dim_value in dims.items():
            cur_prompt = self._fill_prompt(dim_value, source, candidates, reference)
            response, response_count = self._make_reqeuest(prompt=cur_prompt)
            result[dim_name] = self._calucate_score(response, response_count)
        return result