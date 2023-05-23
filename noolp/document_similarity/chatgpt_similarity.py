from typing import List

import numpy as np

from noolp.document_similarity.doc_similarity import DocSimilarity


class ChatGPTSimilarity(DocSimilarity):
    def __init__(
        self,
        openai_key: str,
        documents: List[str],
        language: str = "english",
        verbose=False,
    ):
        super().__init__(documents, language, verbose)

    def get_similarity(self) -> List[List[float]]:
        """"""

        return [[]]
