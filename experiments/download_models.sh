DIR_LIST="0102_L 0103_R 0104_L 0105_R"
for dir in $DIR_LIST
do
  mkdir $dir
  mkdir $dir/models
  wget https://github.com/nsavinov/SPTM_data/raw/master/$dir/models/model_weights.h5
  mv model_weights.h5 $dir/models
done
