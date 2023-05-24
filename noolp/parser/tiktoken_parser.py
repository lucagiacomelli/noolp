from typing import List
import logging

from nltk.corpus import stopwords
import nltk
import tiktoken

from noolp.parser.parser_base import ParserBase

logger = logging.getLogger(__name__)


class TiktokenParser(ParserBase):
    """ """

    def __init__(
        self,
        document: str,
        language: str = "english",
        verbose=False,
        model: str = "gpt-3.5-turbo-0301",
    ):
        super().__init__(document, language, verbose)
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def tokenize(
        self,
        max_number_sentences: int = 50,
        max_tokens_per_sentence: int = 500,
    ) -> List[List[int]]:
        """
        Extracts the sentences from the document and, for each sentence, extract its tokens.
        It uses OpenAI tokenizer to extract the tokens.

        NOTE: The newest embeddings model can handle inputs with up to 8191 input tokens,
        so most of the rows would not need any chunking, but this may not be true for all the documents.

        :param max_number_sentences: raise an Exception if the number of sentences in the document is too high
        :param max_tokens_per_sentence: raise an Exception if a sentence in the document is too long
        :return:
        """

        sentences = self.extract_sentences()
        if len(sentences) > max_number_sentences:
            raise RuntimeError(
                f"The document is too long. Maximum number of sentences == {max_number_sentences}"
            )
        tokens_sentences = []
        for sentence in sentences:
            tokens = self.tokenizer.encode(" " + sentence)
            if len(tokens) > max_tokens_per_sentence:
                raise RuntimeError(
                    f"The sentence is too long. Maximum number of tokens == {max_tokens_per_sentence}"
                )
            tokens_sentences.append(tokens)

        return tokens_sentences

    def number_tokens(self):
        tokens = self.tokenize()
        return sum([len(token_list) for token_list in tokens])

    @classmethod
    def encoder_for_model(cls, model: str):
        """
        Returns an encoder to be used with specific model in the OpenAI API. Example 'gpt-4'
        Check the different types of models here: https://github.com/openai/tiktoken/blob/3e8620030c68d2fd6d4ec6d38426e7a1983661f5/tiktoken/model.py
        """
        return tiktoken.encoding_for_model(model)
