from typing import List

from sentence_transformers import SentenceTransformer

from noolp.document_similarity.doc_similarity import DocSimilarity



class EmbeddingSimilarity(DocSimilarity):
    def __init__(
        self,
        documents: List[str],
        language: str = "english",
        verbose=False,
    ):
        super().__init__(documents, language, verbose)

    def get_similarity(self) -> List[List[float]]:
        """
        1. Choose the embedding model: check that the embedding model is trained with similar data.
            If not, find another model or fine-tuned and re-train.
        2. Ensure embedding space is the same for both documents and queries
        3. Convert the documents in embeddings and store in a Vector store (use a library or a Vector DB).



        """
        raise NotImplementedError("Embedding similarity has not been implemented yet")
