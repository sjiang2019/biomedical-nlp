import gensim
from gensim.corpora import WikiCorpus

WIKI_CORPUS_PATH = "/pool1/users/sjiang/thesis/enwiki-latest-pages-articles.xml.bz2"
PRETRAINED_G_NEWS = "/pool1/users/sjiang/thesis/GoogleNews-vectors-negative300.bin"
NEW_MODEL = "/pool1/users/sjiang/thesis/word2vec_model2"
NEW_MODEL_VECTORS = "/pool1/users/sjiang/thesis/word2vec_model_vectors2"

def tune_embeddings():
    # Load google news pretrained vectors
    google_wv = gensim.models.KeyedVectors.load_word2vec_format(PRETRAINED_G_NEWS, binary=True)
    # Initialize new model
    model = gensim.models.Word2Vec(size=300, min_count=5, iter=10)
    # Transfer wiki corpus vocab to new model
    wiki = WikiCorpus(WIKI_CORPUS_PATH)
    model.build_vocab(wiki.get_texts())
    # Add google new vocab and transfer weights
    training_examples_count = model.corpus_count
    model.build_vocab([list(google_wv.vocab.keys())], update=True)
    model.intersect_word2vec_format(PRETRAINED_G_NEWS, binary=True, lockf=1.0)
    # Train on wiki corpus
    model.train(wiki.get_texts(), total_examples=training_examples_count, epochs=model.iter)
    model.save(NEW_MODEL)
    model.wv.save(NEW_MODEL_VECTORS)

if __name__ == '__main__':
    tune_embeddings()