import os
root = '../../../../../'

os.chdir(root)
#from experiments.common_tools.cmc import evaluateCMC
print 'current work dir : %s'%os.getcwd()
caffe_root = 'caffe'
commontool_root = 'experiments_py/common_tools'
datalayer_root = 'experiments_py/exper_cuhk/cuhk01/'
import sys
sys.path.insert(0,caffe_root+'/python')
sys.path.insert(0,commontool_root)
sys.path.insert(0,datalayer_root)
import caffe
from cuhk01_SeqIn import *
from cmc_cuhk01_tools import *
from cmc_tools import *
import numpy as np
import time
import scipy.io as sio



DATA_DIR='dataset/cuhk01/cuhk01/'

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

save_root = 'experiments_py/exper_cuhk/cuhk01/singleNet/CMC_test/'

save_path = save_root + 'cuhk01_id486_set02_1209_funtune.png'
save_mat_path = save_root + 'CMC_data(single)_set02_1209_funtu.mat'
fig_legend = 'orignal_singlecmc'

DEPLOY_PATH= 'experiments_py/exper_cuhk/cuhk01/singleNet/deploy.prototxt'
LSTM_MODELPATH = 'models/cuhk01/singleNet/funtun_set02_iter_100000.caffemodel'




test_id_number = 486
set_no = 2
phase = 'test'
filename_test = 'dataset/cuhk01/exp_set/testid%03d_set%02d_%s.txt'%(test_id_number,(set_no),phase)
print filename_test



[name_dict, key_list] = parse_dataset(DATA_DIR, filename_test)



    

caffe.set_device(device_id)
caffe.set_mode_gpu()
net = caffe.Net(DEPLOY_PATH, LSTM_MODELPATH, caffe.TEST)

start_time = time.time()
total_time = 0
cmcDict={}
cmc_list=[]

for i in range(round_num):
    print 'Round %d with rand list:'%i
    key_list_shuffle = key_list[:]
    random.shuffle(key_list_shuffle)
    probes, gallerys=readList(key_list_shuffle[:],name_dict,False)
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
        