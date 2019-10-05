import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors
import codecs
import jieba
import numpy as np

infile = './train_brand.txt'
outfile = './word_embedding_train_2.txt'
descsFile = codecs.open(infile,'rb',encoding = 'utf-8')
with open (outfile ,'w',encoding = 'utf-8') as f:
    for line in descsFile:
        line = line.strip()
        line = line.replace('8','')
        words = jieba.cut(line)
        for word in words:
            f.write(word + ' ')
        f.write('\n')

inp = outfile
outp2 = './red_book_word_vector'
model = Word2Vec(LineSentence(inp),size = 200  ,window = 5, min_count = 0,workers = multiprocessing.cpu_count())
model.wv.save_word2vec_format(outp2,binary = False)