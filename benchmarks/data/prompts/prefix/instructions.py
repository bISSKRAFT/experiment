LITERATURE_REVIEW = """
Your literature review should be structured as follows:

### Introduction
Introduce the research question, explain the purpose and scope of your review, and provide an overview of the main themes or subtopics that you can extract from the sources.

### Body
Discuss each theme or subtopic in detail, using evidence and examples from the sources. Summarize the main points, findings, and implications of the sources and identify the similarities, differences, gaps, and contradictions among them. Highlight the strengths and weaknesses of the existing research and how it contributes to the research question.

### Conclusion
Summarize the main findings and implications of your review, identify the limitations and gaps in the literature, and suggest directions for future research.

Provide proper citations in your literature review (only Author & Year).
"""

QA_PORTION="""
Use the following portion of a long document to see if any of the text is relevant to answer the question.
Return any relevant text verbatim, while also providing the document and chunk IDs for each point mentioned."
{summaries}
Question: {question}
Relevant text, if any:"""