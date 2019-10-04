NLP_Keams
==============
This project mainly introduces clustering in the NLP direction.

**********

## Method:Embedding(Tf-idf) + fbpca + Kmeans
### (1)文本向量化
#### I.tf-idf向量化
```Python
cv = TfidfVectorizer(max_df = 100,min_df = 10,binary=False, decode_error='ignore', stop_words=stop_list,)
vec = cv.fit_transform(line_list)
tf_id_vec = vec.toarray()
```
- 其中需要一个stop_list为停词表，停词不作为feature。max_df和min_df分别控制feature的上下线，超过线就会丢掉该词，不进入feature_list，若不加入该参数会导致内存撑爆。
- line_list为文本列表，tf_id_vec为做完特征的array
- 内含逻辑为：整个数据集计算tf-idf生成feature.txt，每个数据从feature.txt里寻找feature寻找不到就是0，寻找到了就用tf-idf的权值作为权重。一般该方法生成的vector是稀疏的。

#### II.Embedding向量化
```Python
model_char = KeyedVectors.load_word2vec_format('./char_vector')
model_word = KeyedVectors.load_word2vec_format('./word_vector')
```
通过gensim的包载入提前训练好的embedding，在此特征为char_embedding和word_embedding，利用gensim中的：
```Python
Word2Vec(LineSentence(inp), size = 200, window = 5, min_count = 0, workers = multiprocessing.cpu_count())
model.wv.save_word2vec_format(outp2, binary = False)
```
可以将模型保存下来。
- 将一句话进行切词，寻找word在model中的权重，累加求平均值。char_embedding做同样的操作，最后connect起来作为最后向量vector。
- Word2Vec的model可以利用dict方法寻找权值。```model['word']```
- 增加手工特征：可以添加手工特征优化vector，比如最终vecter添加unknow_word的计数

### (2)PCA压缩维度
#### I.PCA代码
结束文本向量化内容之后，需要对文本进行降维运算，这样才可以以更少维的vector携带更多的样本信息，对后期聚类运算有更好的效果，倘若不做降维运算并不能将聚类距离清晰的划分开。  
在此选用的是fbPCA（Fast Randomized PCA），选择fbPCA的原因是对于Tf-idf的稀疏矩阵有良好的效果，能正确的携带样本信息。
```Python
(U,s,Va) = fbpca.pca(train_array) #二维矩阵，shape = (batch_size, features)
PCA_list = np.dot(train_array,Va.T)
```
- 核心步骤，不进行该步骤的话，会导致特征不明确，利用欧式距离计算分不开类簇，导致不精准。PCA可以浓缩特征使特征携带整个数据更多的信息。
#### II.PCA和rPCA的区别
M为数据矩阵，L为低秩矩阵，S为稀疏矩阵，N为高斯噪声矩阵。PCA与rPCA的优化目标不同。

    PCA优化目标为：M = L + N
    rPCA优化目标为：M = L + S

PCA可以通过SVD的方法得到L的低秩矩阵，主成分奇异值分解得到的左右奇异阵，进行低秩矩阵的压缩。  
rPCA本质是采用ALM算法对rPCA进行求解

### (3)Kmeans
聚类采用Python中sklearn中的Kmeans，其优点在于简便，其缺点在于不可更改距离公式，大部分功能需要自己实现。返回参数为：

|参数|内容|
|------|------|
|cluster_centers_|返回聚类中心labels和features|
|labels_|返回每个数据点所属于类的编号|
|inertia_|每个类的内簇距离|
|n_iter_|迭代次数|

```Python
cluster = KMeans(n_clusters = n_clusters)
cluster1 = cluster.fit(keams_train_list)
```
- 利用内簇距离投票的方式选择聚簇个数，遍历聚类个数，若其中一个内簇距离小于一个阈值，count+=1，当整体达标类簇百分比大于一个阈值时，选择该类簇。该变化为线性变化，随着聚类个数越多，达标百分比越高（原因是类簇内元素少）。
- 最终做内簇手工特征选择特点类簇。
- 聚类结果可以利用：```cluster = joblib.load('./cluster')```保存。









