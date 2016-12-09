import os
root = '../../../../../'

os.chdir(root)
#from experiments.common_tools.cmc import evaluateCMC
print 'current work dir : %s'%os.getcwd()
caffe_root = 'caffe'
commontool_root = 'experiments_py/common_tools'
datalayer_root = 'experiments_py/exper_cuhk/cuhk03/'
import sys
sys.path.insert(0,caffe_root+'/python')
sys.path.insert(0,commontool_root)
sys.path.insert(0,datalayer_root)
import caffe
from cuhk03_SeqIn import *
from cmc_cuhk03_tools import *
from cmc_tools import *
import numpy as np
import time
import scipy.io as sio



DATA_DIR='dataset/cuhk03/data/'

device_id = int(sys.argv[1])
batch_size = int(sys.argv[2])
round_num = int(sys.argv[3])

frames_l = 1
chnnels = 3
height = 160
width = 80
path_root = './'
multylabel = False
affine = False
balance = False

save_root = 'experiments_py/exper_cuhk/cuhk03/singleNet/CMC_test/'

save_path = save_root + 'cuhk03_set02_1209_funtune.png'
save_mat_path = save_root + 'CMC_data(single)_set02_1209.mat'
fig_legend = 'orignal_singlecmc'

DEPLOY_PATH= 'experiments_py/exper_cuhk/cuhk03/singleNet/deploy.prototxt'
LSTM_MODELPATH = 'models/cuhk03/singleNet/set02_iter_500000.caffemodel'



set_no = 2
phase = 'test'
filename_test = 'dataset/cuhk03/exp_set/set%02d_%s_noval.txt'%((set_no),phase)
print filename_test
num_listfile = 'dataset/cuhk03/split/totalNumListDict.txt'
    

caffe.set_device(device_id)
caffe.set_mode_gpu()
net = caffe.Net(DEPLOY_PATH, LSTM_MODELPATH, caffe.TEST)

start_time = time.time()
total_time = 0
cmcDict={}
cmc_list=[]

for i in range(round_num):
    print 'Round %d with rand list:'%i
    probes,gallerys = readList(filename_test,num_listfile, DATA_DIR, frames_l, balance )
    print len(probes.keys())
    #print probes
    #print gallerys
    key_probe = probes.keys()
    key_gallerys = gallerys.keys()
#     print key_probe
#     print key_gallerys
    
    video_instance = videoReadInput()

    video_instance.initialize('test', False, batch_size, frames_l, 3, 160, 80, path_root,'./','./',multylabel,affine,balance,
                          [probes,key_probe],[gallerys,key_gallerys])

    predictLists=generatePredictList_Batch(net,video_instance,
                                                            [probes,key_probe],[gallerys,key_gallerys],frames_l, batch_size)
    #print predictLists
    #print key_probes
    #print key_gallerys
    cmc=evaluateCMC(predictLists, key_probe, key_gallerys)
    #cmc_max=evaluateCMC(predictLists_max, key_probes, key_gallerys)

    cmc_list.append(cmc)
    #cmc_list_max.append(cmc_max)
    end_time = time.time()
    using_time = end_time - start_time
    total_time = total_time + using_time
    start_time = end_time
    time1 = format_time(long(total_time))
    print 'Time ellipsing: %d days %d hours %d mins %d seconds \n' % (
        time1['days'], time1['hours'], time1['min'], time1['sec'])
    print '%d rounds has been tested' % i
    #np.save('predictLists.npy', predictLists)
    #np.save('key_probes.npy', key_probes)
    #np.save('key_gallerys.npy', key_gallerys)
end_time = time.time()
using_time = end_time - start_time
total_time = total_time + using_time
start_time = end_time
time1 = format_time(long(total_time))
print 'Time ellipsing: %d days %d hours %d mins %d seconds \n' % (
        time1['days'], time1['hours'], time1['min'], time1['sec'])


aver_cmc = np.average(cmc_list[:],axis=0)
#aver_cmc_max = np.average(cmc_list[:],axis=0)
cmcDict[fig_legend] = aver_cmc
#cmcDict['ours (max_pooling)'] = aver_cmc_max
    

    
#cmcDict={}

plotCMC(cmcDict,save_path)
sio.savemat(save_mat_path,{'last':aver_cmc})
        