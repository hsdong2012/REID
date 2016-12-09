DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

cd ../../../../
export PYTHONPATH=./experiments_py/exper_cuhk/cuhk01/:$PYTHONPATH

caffe/build/tools/caffe train -solver experiments_py/exper_cuhk/cuhk01/singleNet/solver.prototxt -gpu $1 -weights models/cuhk01/singleNet/set02_iter_320000.caffemodel