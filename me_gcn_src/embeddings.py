import numpy as np
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def generate_word_embeddings(word_list, tokenized_docs):
    wv_cbow_model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=25,
        window=5,
        min_count=0,
        epochs=100,
    )
    word_emb_dict = {}
    wv = wv_cbow_model.wv
    for word in word_list:
        word_emb_dict[word] = wv[word].tolist()

    return word_emb_dict


def generate_doc_embeddings(tokenized_docs):
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenized_docs)]
    model = Doc2Vec(
        documents, vector_size=25, window=5, min_count=1, workers=4, epochs=200
    )

    doc2vec_emb = []
    for i in range(len(documents)):
        doc2vec_emb.append(model.docvecs[i])
    doc2vec_npy = np.array(doc2vec_emb)
    return doc2vec_npy
