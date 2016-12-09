from math import *
import random
import sys

from experiments_py.common_tools.base_input_layer import videoRead
from experiments_py.common_tools.ProgressBar import *
# def progresslog(s,i):
#     sys.stdout.write(s+':{0:3}/{1:3}:'.format(i, 100)+'#'*i+'->'+'\r')
def readlistFromFile(video_list,frames_l,balance,height,width,multylabel):

    f = open(video_list, 'r')
    f_lines = f.readlines()
    # print f_lines
    f.close()
    
    video_dict = {}
    current_line = 0
    video_order = []
    c = 0
    total = len(f_lines)
    point = total/100
    for ix, line in enumerate(f_lines):
        videokey = ix
        video_dict[videokey] = {}
        video_dict[videokey]['frames'] = []
        
        video_dict[videokey]['frames_p'] = []

        filename = line.split(' ')[0]
        filepre = filename[:-7]+'%03d'+filename[-4:]
        person_id = int(filename.split('/')[-1][:-7])
        excluId = int(filename[-7:-4])
        if frames_l>1 :
            for pid in range(4):
                if (pid+1) != excluId:
                    if balance == True:
                        video_dict[videokey]['frames'].append(filepre%(pid+1))  # should verify  
                    else:
                        video_dict[videokey]['frames'].append(filename)
        else:
             video_dict[videokey]['frames'].append(line.split(' ')[0])
        
        filename = line.split(' ')[1]
        person_idp = int(filename.split('/')[-1][:-7])
        filepre = filename[:-7]+'%03d'+filename[-4:]
        excluId = int(filename[-7:-4])
        if frames_l>1:
            for pid in range(4):
                if (pid+1) != excluId:
                    video_dict[videokey]['frames_p'].append(filepre%(pid+1))  # should verify
        else:
             video_dict[videokey]['frames_p'].append(line.split(' ')[1])
        



        video_dict[videokey]['reshape'] = (height, width)
        video_dict[videokey]['crop'] = (height, width)

        video_dict[videokey]['label'] = []
        if multylabel == True:
            video_dict[videokey]['label'].append(int(line.split(' ')[2]))
            video_dict[videokey]['label'].append(person_id)
            video_dict[videokey]['label'].append(person_idp)

        else:
            video_dict[videokey]['label'].append(int(line.split(' ')[2]))

        video_order.append(videokey)
        if ix % point == 0:
            progresslog('data list is loading',ix/point)

    print 'video_list:%s'%(video_list)
    print 'list eaxample:'
    print random.choice(video_dict)
    return [video_dict,video_order]




#Train_batch_size = 100
#Test_batch_size = 10
#Train_frames_l = 1 #SEQUENCE LENGTH
#Test_frames_l = 1
chnnels = 3
height = 160
width = 80
path_root = './'

TrainVideolist ='dataset/cuhk01/split/img_seq/testid486_set02_train_pair.txt'
TestVideolist ='dataset/cuhk01/split/img_seq/testid486_set02_test_pair.txt'
TrainVideolistFlow =''
TestVideolistFlow =''
affine = True
#multylabel = False
#balance = False #if True then seq-seq;else then img-seq





class videoReadTrain_flow(videoRead):

  def initialize(self):
    params = eval(self.param_str)
    Train_batch_size = int(params['Train_batch_size'])
    Train_frames_l = int(params['Train_frames_l'])
    multylabel = bool(params['multylabel']=='True')
    balance = bool(params['balance']=='True')
    self.train_or_test = 'train'
    self.flow = True
    self.buffer_size = Train_batch_size  #num videos processed per batch
    self.frames = Train_frames_l   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.idx = 0
    self.channels = chnnels
    self.height = height
    self.width = width
    self.path_to_images = path_root  # the pre path to the datasets
    self.multylabel = multylabel
    self.affine = affine
    video_dict = readlistFromFile(TrainVideolistFlow,self.frames,balance,self.height,self.width,self.multylabel)
    self.video_dict = video_dict[0]
    self.video_order = video_dict[1]
    self.num_videos = len(self.video_dict)

class videoReadTest_flow(videoRead):

  def initialize(self):
    params = eval(self.param_str)
    Test_batch_size = int(params['Test_batch_size'])
    Test_frames_l = int(params['Test_frames_l'])
    multylabel = bool(params['multylabel']=='True')
    balance = bool(params['balance']=='True')
    self.train_or_test = 'train'
    self.flow = True
    self.buffer_size = Test_batch_size  #num videos processed per batch
    self.frames = Test_frames_l   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.idx = 0
    self.channels = chnnels
    self.height = height
    self.width = width
    self.path_to_images = path_root  # the pre path to the datasets
    self.multylabel = multylabel
    self.affine = affine
    video_dict = readlistFromFile(TestVideolistFlow,self.frames,balance,self.height,self.width,self.multylabel)
    self.video_dict = video_dict[0]
    self.video_order = video_dict[1]
    self.num_videos = len(self.video_dict)
    

class videoReadTrain(videoRead):

  def initialize(self):
    params = eval(self.param_str)
    Train_batch_size = int(params['Train_batch_size'])
    Train_frames_l = int(params['Train_frames_l'])
    multylabel = bool(params['multylabel']=='True')
    balance = bool(params['balance']=='True')
    self.train_or_test = 'train'
    self.flow = False
    self.buffer_size = Train_batch_size  #num videos processed per batch
    self.frames = Train_frames_l   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.idx = 0
    self.channels = chnnels
    self.height = height
    self.width = width
    self.path_to_images = path_root  # the pre path to the datasets
    self.multylabel = multylabel
    self.affine = affine
    video_dict = readlistFromFile(TrainVideolist,self.frames,balance,self.height,self.width,self.multylabel)
    self.video_dict = video_dict[0]
    self.video_order = video_dict[1]
    self.num_videos = len(self.video_dict)

class videoReadTest(videoRead):

  def initialize(self):
    params = eval(self.param_str)
    Test_batch_size = int(params['Test_batch_size'])
    Test_frames_l = int(params['Test_frames_l'])
    multylabel = bool(params['multylabel']=='True')
    balance = bool(params['balance']=='True')
    self.train_or_test = 'train'
    self.flow = False
    self.buffer_size = Test_batch_size  #num videos processed per batch
    self.frames = Test_frames_l  #length of processed clip
    self.N = self.buffer_size*self.frames
    self.idx = 0
    self.channels = chnnels
    self.height = height
    self.width = width
    self.path_to_images = path_root  # the pre path to the datasets
    self.multylabel = multylabel
    self.affine = affine
    video_dict = readlistFromFile(TestVideolist,self.frames,balance,self.height,self.width,self.multylabel)
    self.video_dict = video_dict[0]
    self.video_order = video_dict[1]
    self.num_videos = len(self.video_dict)
