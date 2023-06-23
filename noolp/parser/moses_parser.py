from typing import List
import logging

from datasets import load_dataset
from transformers import pipeline

from noolp.parser.parser_base import ParserBase

logger = logging.getLogger(__name__)


class MosesParser(ParserBase):
    """ """

    def __init__(
        self,
        document: str,
        language: str = "english",
        verbose=False,
    ):
        super().__init__(document, language, verbose)
        # self.tokenizer = tiktoken.encoding_for_model(model)

    def tokenize(
        self,
        max_number_sentences: int = 50,
        max_tokens_per_sentence: int = 500,
    ) -> List[List[int]]:
        """

        TODO: implement this function

        :param max_number_sentences: raise an Exception if the number of sentences in the document is too high
        :param max_tokens_per_sentence: raise an Exception if a sentence in the document is too long
        :return:
        """

        raise NotImplementedError()

    def number_tokens(self):
        tokens = self.tokenize()
        return sum([len(token_list) for token_list in tokens])
