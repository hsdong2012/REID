DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

cd ../../../
export PYTHONPATH=./experiments_py/exper_prid/:$PYTHONPATH

caffe/build/tools/caffe train -solver experiments_py/exper_prid/LstmNet/lstm_solver_RGB.prototxt -gpu $1 -weights models/googlenet/bvlc_googlenet.caffemodel