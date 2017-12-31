from konlpy.tag import Mecab
import numpy as np
import os
import argparse
# import h5py # conda install -c anaconda h5py
# import pickle

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot


def save_sentences(name):
    with open('./data/news/' + name + '.txt', 'r') as f:
        lines = f.read()
        sentences = lines.split('.')
        sentences = list(map(lambda x: mecab.morphs(x), sentences[:20]))
        sentences = np.array(sentences)
        # with h5py.File("./data/news/" + name + ".hdf", "w") as outfile:
        #     dset = outfile.create_dataset(name, data=sentences, chunks=True)
        # with gzip.open('./data/news/' + name + '.dmp', "wb") as ft:
        #     pickle.dump(sentences, ft)
        np.save('./data/news/'+ name + '.npy', sentences)

        print("complete saving " + name)

def load_sentences(name):
    # f = h5py.File('./data/news/' + name + '.hdf', 'r')
    # return f[name]
    # with gzip.open('./data/news/' + name + '.dmp', 'rb') as f:
    #     loaded_f = pickle.load(f)
    loaded_f = np.load('./data/news/'+ name + '.npy')
    print("complete loading " + name)
    return loaded_f

def make_word2vec(trained, size, iter, sg):
    if trained == 'n':
        articles = []
        for news_name in ['chosun_full', 'donga_full', 'hani_full', 'joongang_full', 'kh_full'] :
            if os.path.exists('./data/news/' + news_name + '.npy'):
                data = np.array(load_sentences(news_name))
                articles.append(data)
            else :
                save_sentences(news_name)

        for sentences in articles:
            if os.path.exists('./word2vec'):
                model = Word2Vec.load('./word2vec')
                model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
            else :
                model = Word2Vec(sentences, min_count=15, size=size, workers=3, sg=sg)
                model.save('./word2vec')
        print("complete training model")
    else :
        model = Word2Vec.load('./word2vec')
        print("complete loading model")
        print("model.wv['박근혜']",model.wv['박근혜'])
        print("model.wv['이명박']",model.wv['이명박'])
        print("model.wv.similarity('이명박', '박근혜')", model.wv.similarity('이명박', '박근혜'))
        print("model.wv.similarity('문재인', '박근혜')", model.wv.similarity('문재인', '박근혜'))
        print("model.wv.most_similar(positive=['박근혜'], topn=1", model.wv.most_similar(positive=['박근혜'], topn=1))
        print("model.wv.most_similar(positive=['문재인'], topn=1", model.wv.most_similar(positive=['문재인'], topn=1))




def main() :
    parser = argparse.ArgumentParser(description='This code is transforming news_data to vector-from')
    parser.add_argument('--trained', type=str, default='y',\
                        help="if you have already trained model, insert 'y' \n \
                              or if you don't, insert 'n'")
    parser.add_argument('--size', type=int, default=100,\
                        help="list of embedding sizes. e.g. [100, 200, 300]")
    parser.add_argument('--iter', type=int, default=10,\
                        help="list of # iteration e.g. [10, 20,")
    parser.add_argument('--sg', type=int, default=0,\
                        help= "list of model type e.g. [0]"
                        "CBOW have sg value 0, otherwise like skip-gram has sg value 1")


    args = parser.parse_args()
    trained = args.trained
    size = args.size
    iter = args.iter
    sg = args.sg

    mecab = Mecab()
    make_word2vec(trained, size, iter, sg)

if __name__=="__main__":
    main()
