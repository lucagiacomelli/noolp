from abc import abstractmethod


class TopicModeller:
    def __init__(
        self,
        name,
        verbose: bool = False,
    ):
        self.name = name
        self.verbose = verbose

    @abstractmethod
    def extract_topics(self, story, use_lemmatization: bool = True):
        pass
