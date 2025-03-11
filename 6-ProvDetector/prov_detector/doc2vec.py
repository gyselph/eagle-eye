"""Learn a Doc2Vec embedding. For our use case, each document is one rare path, represented as one long string."""

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

WORKERS = 16
WORD_SEPERATOR = " ||| "


def train_doc2vec(sentences: np.ndarray, embedding_dim: int, epochs: int) -> Doc2Vec:
    """
    Train a doc2vec model, using the Gensim library.

    :param sentences: The rare paths in provenance graphs
    :param embedding_dim: The traget dimension of the Doc2Vec embedding
    :param epochs: The number of training epochs
    :return: The trained Doc2Vec model
    """
    tokenized_sentences = _preprocess_sentences(sentences)
    print(f"Starting Doc2Vec training on {len(tokenized_sentences)} data points ...")
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenized_sentences)]
    model = Doc2Vec(documents, vector_size=embedding_dim, workers=WORKERS, epochs = epochs, dm = 1)
    return model


def embed_doc2vec(doc2vec_model: Doc2Vec, raw_sentences: np.ndarray) -> np.ndarray:
    """
    Embed senteces using a trained Doc2Vec model.

    :param doc2vec_model: The trained model
    :param raw_sentences: The rare paths from provenance graphs
    :return: The sentence embeddings, which will be of shape (number of sentences, embedding_dim)
    """
    tokenized_sentences = _preprocess_sentences(raw_sentences)
    print("Embedding {} sentences via Doc2Vec ...".format(len(tokenized_sentences)))
    embeddings = [doc2vec_model.infer_vector(line) for line in tokenized_sentences]
    embeddings = np.stack(embeddings, axis=0)
    return embeddings


def _preprocess_sentences(sentences):
    tokenized_sentences = [line.split(WORD_SEPERATOR) for line in sentences]
    return tokenized_sentences