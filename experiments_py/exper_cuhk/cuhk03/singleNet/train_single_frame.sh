DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

cd ../../../../
export PYTHONPATH=./experiments_py/exper_cuhk/cuhk03/:$PYTHONPATH

caffe/build/tools/caffe train -solver experiments_py/exper_cuhk/cuhk03/singleNet/solver.prototxt -gpu $1 -weights models/googlenet/bvlc_googlenet.caffemodel
