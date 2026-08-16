import os

from sentence_transformers import SentenceTransformer

# bge-small keeps the 512-token window the 1500-char chunks need while being
# ~10x smaller than bge-large, which matters on a low-memory machine.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", 16))

# bge models are trained to embed queries with this instruction prefix; passages
# are embedded without it.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def get_embedding_function():
    model = SentenceTransformer(EMBED_MODEL)

    class FinanceEmbeddings:
        def __init__(self, model):
            self.model = model

        def _encode(self, texts):
            # bge expects cosine similarity over normalized vectors.
            return self.model.encode(
                texts,
                batch_size=EMBED_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

        def embed_documents(self, texts):
            return self._encode(list(texts))

        def embed_query(self, text):
            return self._encode([QUERY_PREFIX + text])[0]

    return FinanceEmbeddings(model)
