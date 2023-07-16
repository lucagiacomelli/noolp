from typing import List

from sentence_transformers import SentenceTransformer, util

from noolp.document_similarity.doc_similarity import DocSimilarity


class EmbeddingSimilarity(DocSimilarity):
    """
    Models: 'all-MiniLM-L6-v2', 'distilbert-base-nli-stsb-mean-tokens'
    """

    def __init__(
        self,
        documents: List[str],
        model: str = "all-MiniLM-L6-v2",
        language: str = "english",
        verbose=False,
    ):
        super().__init__(documents, language, verbose)
        self.model = SentenceTransformer(model)

    def get_similarity(self) -> List[List[float]]:
        """
        1. Choose the embedding model: check that the embedding model is trained with similar data.
            If not, find another model or fine-tuned and re-train.
        2. Ensure embedding space is the same for both documents and queries
        3. Convert the documents in embeddings and store in a Vector store (use a library or a Vector DB).

        TODO: change from sentence_transformers to hugging face transformers,
          because the sentence_transformers package seems not not be maintained anymore
        """

        # Compute embeddings
        embeddings = self.model.encode(self.documents, convert_to_tensor=True)

        # Compute cosine-similarities for each sentence with each other sentence
        cosine_scores = util.cos_sim(embeddings, embeddings)
        return cosine_scores.numpy()
