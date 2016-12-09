import os

root = '../../../'

os.chdir(root)
print 'current work dir : %s' % os.getcwd()
caffe_root = 'caffe'
commontool_root = 'experiments_py/common_tools'
import sys

sys.path.insert(0, caffe_root + '/python')
sys.path.insert(0, commontool_root)
import caffe
from experiments_py.exper_ilids.ilids_SeqIn import *
from experiments.common_tools.cmc_tools import *
import numpy as np
import time

# now begin trainning and testing the net

# setting the pre parameters

Train_batch_size = 12
Test_batch_size = 12
Train_frames_l = 3 #SEQUENCE LENGTH
Test_frames_l = 3
chnnels = 3
height = 160
width = 80
path_root = './'

TrainVideolist ='dataset/i-LIDS-VID/split/img_seq/sequences_set01_train.txt'
TestVideolist ='dataset/i-LIDS-VID/split/img_seq/sequences_set01_train.txt'
Num_list = 'dataset/i-LIDS-VID/split/totalNumListDict.txt'
TrainVideolistFlow =''
TestVideolistFlow =''
multylabel = False
affine = True
balance = False


video_instance = videoReadInput()
video_testInstance = videoReadInput()

video_instance.initialize('train', False, Train_batch_size, Train_frames_l, 3, 160, 80, path_root, TrainVideolist,Num_list,multylabel,affine,balance)
video_testInstance.initialize('test', False, Test_batch_size , Test_frames_l, 3, 160, 80, path_root, TestVideolist,Num_list,multylabel,affine,balance)
video_instance.setup()
video_testInstance.setup()

SOLVER_PATH = 'experiments_py/exper_ilids/LstmNetTest/lstm_solver_RGB.prototxt'
caffe.set_device(0)
caffe.set_mode_gpu()
### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.SGDSolver(SOLVER_PATH)

solver.net.copy_from('models/googlenet/bvlc_googlenet.caffemodel')

log_file = 'lo.txt'
try:
    fp = open(log_file, 'w')
except IOError, e:
    print "IOError"
max_iter = 80000
test_iter = 100
test_interval = 1000
snapshot = 5000
display = 200
# losses will also be stored in the log
train_loss = np.zeros(max_iter)
test_acc = np.zeros(int(np.ceil(max_iter / test_interval)))

start_time = time.time()
total_time = 0

# the main solver loop
for it in range(max_iter):

    video_instance.forward(solver.net)
    solver.step(1)  # SGD by Caffe
    # video_instance.forward(solver.net)
    # solver.step(1)  # SGD by Caffe

    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    accuracy = solver.net.blobs['accuracy'].data
    # print solver.net.blobs['label'].data[:8]
    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        accu = 0
        for test_it in range(test_iter):
            video_testInstance.forward(solver.test_nets[0])
            solver.test_nets[0].forward()
            accu += solver.test_nets[0].blobs['accuracy'].data
        test_acc[it // test_interval] = accu / test_iter
        print test_acc[it // test_interval]
    # output the log
    if (it % test_interval == 0) and (it % display == 0):
        head = 'Iteration ' + '%d,' % (it) + '\n'
        fp.write(head)
        col1 = 'Train net output: accuracy = %f; loss = %f; \n' % (accuracy, train_loss[it])
        fp.write(col1)
        col2 = 'Test net output: accuracy = %f; \n' % (test_acc[it // test_interval])
        fp.write(col2)
        end_time = time.time()
        using_time = end_time - start_time
        total_time = total_time + using_time
        start_time = end_time
        time1 = format_time(long(total_time))
        col3 = 'Time ellipsing: %d days %d hours %d mins %d seconds \n' % (
        time1['days'], time1['hours'], time1['min'], time1['sec'])
        fp.write(col3)
        fp.flush()

    if it % snapshot == 0 and it > 1:
        solver.snapshot()
fp.close()
from pylab import *

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(max_iter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')







