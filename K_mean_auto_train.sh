file_path=$1
rm -r cluster_dir/train_cluster_dir
mkdir cluster_dir/train_cluster_dir
cat ${file_path} | python Kmeans_.py 'embadding' 'train' None
cd cluster_dir
python Choice_h_train.py train_cluster_dir

