# NLP_Keams
This project mainly introduces clustering in the NLP direction.

## Method:Embedding(Tf-idf) + fbpca + Kmeans
### (1)文本向量化
I.tf-idf向量化
```Python
cv = TfidfVectorizer(max_df = 100,min_df = 10,binary=False, decode_error='ignore', stop_words=stop_list,)
vec = cv.fit_transform(line_list)
tf_id_vec = vec.toarray()
```
  其中需要一个stop_list为停词表，停词不作为feature。max_df和min_df分别控制feature的上下线，超过线就会丢掉该词，不进入feature_list，若不加入该参数会导致内存撑爆。
  line_list为文本列表，tf_id_vec为做完特征的array
	内含逻辑为：整个数据集计算tf-idf生成feature.txt，每个数据从feature.txt里寻找feature寻找不到就是0，寻找到了就用tf-idf的权值作为权重。一般该方法生成的vector是稀疏的。
