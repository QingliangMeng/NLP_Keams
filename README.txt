训练cluster
cat ad_sample.txt | python Kmeans_.py tf-idf train none
cat ad_sample.txt | python Kmeans_.py embadding train none

测试
cat black_sample.txt | python Kmeans_.py tf-idf test ./output_file.txt
cat black_sample.txt | python Kmeans_.py embadding test ./output_file.txt

输出文件代表选择里聚类中心最远的几条样本，几条样本可以调整控制
注意测试时保证当前文件目录下cluster文件已经按照方法更新
