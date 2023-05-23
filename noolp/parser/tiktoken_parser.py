from typing import List

from nltk.corpus import stopwords
import nltk
import tiktoken

from noolp.parser.parser_base import ParserBase


class TiktokenParser(ParserBase):
    """ """

    def __init__(
        self,
        document: str,
        language: str = "english",
        verbose=False,
        encoding: str = "cl100k_base",
    ):
        super().__init__(document, language, verbose)
        # By default, load the cl100k_base tokenizer which is designed to work with the ada-002 model
        self.tokenizer = tiktoken.get_encoding(encoding)

    def tokenize(
        self,
        include_stop_words: bool = True,
        include_punctuation: bool = True,
        max_number_sentences: int = 50,
        max_tokens_per_sentence: int = 500,
    ) -> List[List[str]]:
        """
        Extracts the sentences from the document and, for each sentence, extract its tokens.
        It uses OpenAI tokenizer to extract the tokens.

        NOTE: The newest embeddings model can handle inputs with up to 8191 input tokens,
        so most of the rows would not need any chunking, but this may not be true for all the documents.

        :param include_stop_words: if True, includes the stopwords in a given language.
        :param include_punctuation: if True, includes the punctuation in a given language
        :return:
        """

        stop_words_set = set(stopwords.words(self.language))
        punctuation_set = self._punctuation_set()
        sentences = self.extract_sentences()
        if len(sentences):
            raise RuntimeError(
                f"The document is too long. Maximum number of sentences == {max_number_sentences}"
            )
        tokens_sentences = []

        result = self.tokenizer.encode(self.document)
        print(result)

        for sentence in sentences:

            tokens = self.tokenizer.encode(" " + sentence)
            if len(tokens) > max_tokens_per_sentence:
                raise RuntimeError(
                    f"The sentence is too long. Maximum number of tokens == {max_tokens_per_sentence}"
                )

            print(tokens)

            if not include_stop_words:
                tokens = [token for token in tokens if token not in stop_words_set]
            if not include_punctuation:
                tokens = [token for token in tokens if token not in punctuation_set]
            tokens_sentences.append(tokens)

        return tokens_sentences

    @classmethod
    def encoder_for_model(cls, model: str):
        """Returns an encoder to be used with specific model in the OpenAI API. Example 'gpt-4'"""
        return tiktoken.encoding_for_model(model)
