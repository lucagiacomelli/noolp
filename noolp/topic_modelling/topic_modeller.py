from abc import abstractmethod


class TopicModeller:
    def __init__(
        self,
        name,
        document,
        verbose: bool = False,
    ):
        self.name = name
        self.document = document
        self.verbose = verbose

    @abstractmethod
    def extract_topics(self, use_lemmatization: bool = True):
        pass
