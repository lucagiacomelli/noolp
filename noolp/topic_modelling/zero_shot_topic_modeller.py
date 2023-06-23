from abc import abstractmethod

from noolp.topic_modelling.topic_modeller import TopicModeller


class ZeroShotTopicModeller(TopicModeller):
    def __init__(
        self, name: str, document: str, candidates: list[str], verbose: bool = False
    ):
        super().__init__(name, document, verbose)
        self.candidates = candidates

    @abstractmethod
    def extract_topics(self, use_lemmatization: bool = False):
        raise NotImplementedError()
