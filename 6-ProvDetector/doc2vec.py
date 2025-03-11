from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import os

WORKERS = 16
WORD_SEPERATOR = " ||| "


def train_doc2vec(sentences, doc2vec_model_file, doc2vec_embedding_db, embedding_dim, epochs, overwrite):
    if not overwrite and (os.path.isfile(doc2vec_model_file) or os.path.isfile(doc2vec_embedding_db)):
        print("Doc2Vec embeddings are already present, don't overwrite")
        return load_embeddings(doc2vec_embedding_db)
    model = train_doc2vec_dont_embed(sentences, doc2vec_model_file, embedding_dim, epochs, overwrite)
    tokenized_sentences = preprocess_sentences(sentences)
    embeddings = embed_preprocessed_sentences(tokenized_sentences, model)
    store_embeddings(doc2vec_embedding_db, embeddings)
    return embeddings


def train_doc2vec_dont_embed(sentences, doc2vec_model_file, embedding_dim, epochs, overwrite):
    if not overwrite and os.path.isfile(doc2vec_model_file):
        print("Doc2Vec model already present, don't overwrite")
        return load_model(doc2vec_model_file)
    tokenized_sentences = preprocess_sentences(sentences)
    print("Starting Doc2Vec training on {} data points ...".format(len(tokenized_sentences)))
    model = train_model(tokenized_sentences, embedding_dim, epochs)
    store_model(doc2vec_model_file, model)
    return model


def preprocess_sentences(sentences):
    tokenized_sentences = [line.split(WORD_SEPERATOR) for line in sentences]
    return tokenized_sentences


def train_model(tokenized_sentences, embedding_dim, epochs):
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenized_sentences)]
    model = Doc2Vec(documents, vector_size=embedding_dim, workers=WORKERS, epochs = epochs, dm = 1)
    return model


def embed_preprocessed_sentences(tokenized_sentences, model):
    embeddings = [model.infer_vector(line) for line in tokenized_sentences]
    embeddings = np.stack(embeddings, axis=0)
    return embeddings


def store_embeddings(doc2vec_embedding_db, embeddings):
    print("Storing {} doc2vec embeddings in DB".format(len(embeddings)))
    np.savez_compressed(doc2vec_embedding_db, embeddings = embeddings)


def load_embeddings(doc2vec_embedding_db):
    return np.load(doc2vec_embedding_db)["embeddings"]


def store_model(dec2vec_model_file, model):
    model.save(dec2vec_model_file)


def load_model(dec2vec_model_file):
    return Doc2Vec.load(dec2vec_model_file)


def embed_raw_sentences_doc2vec(dec2vec_model_file, doc2vec_embeddings_file, raw_sentences, overwrite):
    if not overwrite and os.path.isfile(doc2vec_embeddings_file):
        print("Doc2Vec embeddings are already present, don't overwrite")
        return load_embeddings(doc2vec_embeddings_file)
    model = load_model(dec2vec_model_file)
    tokenized_sentences = preprocess_sentences(raw_sentences)
    print("Embedding {} sentences via Doc2Vec ...".format(len(tokenized_sentences)))
    embeddings = embed_preprocessed_sentences(tokenized_sentences, model)
    store_embeddings(doc2vec_embeddings_file, embeddings)
    return embeddings