# -*- coding: utf-8 -*-
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.append('../envs/tensorflow/lib/python3.6/site-packages')
from scipy.spatial.distance import cdist
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from sys import argv
import jieba
import fbpca
import copy
import codecs

np.set_printoptions(threshold=216)
keys_word_method = argv[1]
keys_word_model = argv[2]
output_path = argv[3]
n_clusters = 0

def K_cluster_distance(X,k_center,k_labels,numb_of_cluster):
    return (np.mean(np.sqrt(np.sum((X[k_labels == numb_of_cluster] - k_center[numb_of_cluster])**2,axis = 1))))

def K_choose_standar(X,k_center,k_labels,thorld):
    result = []
    for i in range(0,len(k_center)):
        result.append(np.mean(np.sqrt(np.sum((X[k_labels == i] - k_center[i])**2,axis = 1))))
    return (list(pd.Series(result) < thorld).count(True) / float(len(result)))


def distance_kmeans_center(keams_train_list, cluster):
    res_array = []
    numb_cluster,cluster_feature_len = cluster.cluster_centers_.shape
    for i in range(0, numb_cluster):
        cluster_center_vec = cluster.cluster_centers_[i].reshape(1, cluster_feature_len)
        distance_i = np.sqrt(np.sum((keams_train_list - cluster_center_vec) ** 2, axis=1))
        res_array.append(distance_i)
    res_array = np.array(res_array)
    res_array = np.min(res_array, axis=0)
    return res_array


if keys_word_method == 'tf-idf':
    line_list = []
    stop_list = []
    with open('./stop_word.txt','r') as f:
        for line in f:
            stop_list.append(line.strip())

    for line in sys.stdin:
        line_list.append(line.strip())
    cv = TfidfVectorizer(max_df = 100,min_df = 10,binary=False, decode_error='ignore', stop_words=stop_list,)
    vec = cv.fit_transform(line_list)
    tf_id_vec = vec.toarray()
    #print(tf_id_vec.shape)
    (U,s,Va) = fbpca.pca(tf_id_vec)
    #print(U.shape,s.shape,Va.shape)
    tf_id_pca_vec = np.dot(tf_id_vec,Va.T)
    #print(tf_id_pca_vec.shape)
    keams_train_list = copy.deepcopy(tf_id_pca_vec)
    print(keams_train_list.shape)
    line_list = pd.Series(line_list)


if keys_word_method == 'embadding':
    model_char = KeyedVectors.load_word2vec_format('./char_vector')
    model_word = KeyedVectors.load_word2vec_format('./word_vector')
    keams_train_list = []
    line_list = []
    for line in sys.stdin:
        line = line.strip().decode('utf-8')
        words_values = 0
        char_values = 0
        unknow_char_count = 0
        unknow_words_count = 0
        know_char_count =  0
        know_words_count = 0
        line = line.strip()
        line = line.replace('8','')
        sentence_lenth = len(line)
        if sentence_lenth == 0:
            continue
        words = jieba.cut(line)
        for word in words:
            try:
                words_values += model_word.get_vector(word)
                know_words_count +=1
            except:
                words_values += 0
                unknow_words_count += 1
        for char in list(line) :
            try:
                char_values += model_char.get_vector(char)
                know_char_count += 1
            except:
                char_values += 0
                unknow_char_count += 1
        #print(char_values , words_values)
        if np.sum(char_values) != 0 and np.sum(words_values) != 0:
            sentence_values = list(char_values / know_char_count) + list(words_values / know_words_count)
            sentence_values = sentence_values + [unknow_words_count,unknow_char_count]
            #print(sentence_values)
            keams_train_list.append(sentence_values)
            line_list.append(line)
    line_list = pd.Series(line_list)
    keams_train_list = np.array(keams_train_list)
    print (keams_train_list.shape)
    (U,s,Va) = fbpca.pca(keams_train_list)
    keams_train_list = np.dot(keams_train_list,Va.T)
    print (keams_train_list.shape)




#####选择聚类类簇个数，进行手肘图
if keys_word_model == 'cross':
    print('#--------------cross---------------#')
    mean_distortions = []
    for k in range(1, 20):
        cluster = KMeans(n_clusters = k)
        cluster.fit(keams_train_list)
        mean_distortions.append(sum(np.min(cdist(keams_train_list, cluster.cluster_centers_, metric='euclidean'), axis=1)))
    print(mean_distortions)

#####根据手肘图的聚类个数3进行训练，储存模型
if keys_word_model == 'train':
    print('#---------------train---------------#')
    loop_thorld = 0
    while loop_thorld < 0.8 and n_clusters < 100:
        n_clusters += 5
        cluster = KMeans(n_clusters = n_clusters)
        cluster1 = cluster.fit(keams_train_list)
        #joblib.dump(cluster1,'./cluster')
        numb_cluster,cluster_feature_len = cluster1.cluster_centers_.shape
        loop_thorld = K_choose_standar(keams_train_list,cluster1.cluster_centers_,cluster1.labels_,2)
        print ('当前k为%s,内聚距达标占比%s' %(n_clusters,loop_thorld))
    cluster_label = pd.Series(cluster1.labels_)
    print('最终聚类中心个数%s ' %len(list(set(cluster_label))))
    print('写入每个聚类数据信息，在cluster_dir/train_cluster_dir下')
    for i in range(0,n_clusters):
        cluster_list_index = list(cluster_label[cluster_label == i].index)
        cluster_distance = K_cluster_distance(keams_train_list,cluster1.cluster_centers_,cluster1.labels_,i)
        with codecs.open ('./cluster_dir/train_cluster_dir/' + str(i) + '_cluster','w',encoding='utf-8') as f:
            f.write('cluster_distance = %s' %cluster_distance + '\n')
            for line_index in cluster_list_index:
                f.write(line_list[line_index] + '\n')

#####读取cluster进行预测
if keys_word_model == 'test':
    print('#---------------test---------------#')
    if keys_word_method == 'tf-idf':
        cluster = joblib.load('./cluster_tf_idf')
    if keys_word_method == 'embadding':
        cluster = joblib.load('./cluster_embadding')
    numb_h_train = int(keams_train_list.shape[0] * 0.2)
    if numb_h_train < 500:
        numb_h_train = keams_train_list.shape[0]
    print('h_train样本数量为%s' %numb_h_train)
    res_distance = pd.Series(distance_kmeans_center(keams_train_list,cluster))
    print(np.max(res_distance),np.mean(res_distance),np.median(res_distance))
    h_train = line_list[res_distance.sort_values(ascending = False)[:numb_h_train].index]
    with open(output_path,'w') as f:
        for line in h_train:
            f.write(line.strip() + '\n')
