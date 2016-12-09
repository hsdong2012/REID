import random
import os

import glob
from multiprocessing import Pool
from experiments_py.common_tools.base_input import videoRead
from experiments_py.common_tools.ProgressBar import *
# def progresslog(s,i):
#     sys.stdout.write(s+':{0:3}/{1:3}:'.format(i, 100)+'#'*i+'->'+'\r')
def format_time(num_time):
    seconds=num_time%60
    minuites=num_time/60%60
    hours=num_time/3600%24
    days=num_time/3600/24
    fm_time={'sec':seconds,'min':minuites,'hours':hours,'days':days}
    return fm_time
def processList(filelistname,frames_l,balance,height,width,list_num):
    video_dict = {}
    videokey = int(filelistname.split(' ')[0])
    video_dict[videokey] = {}
    video_dict[videokey]['frames'] = []
    video_dict[videokey]['frames_p'] = []

    filename = filelistname.split(' ')[1]
    filepre = filename.split('/')[-1]
    fileser = filepre[:-10]
    fileTem = filename[:-9]+'%05d'+filename[-4:]
    # print fileTem
    startID = int(filename[-9:-4])
    # print startID
    if frames_l>1 :
        if balance == True:
            # file_list = glob.glob(fileser)
            # file_list.sort()
            file_num = list_num[fileser]
            # print file_num
            endID = startID+ frames_l
            if endID > (file_num[0]+file_num[1]):
                startID = file_num[0]+file_num[1] - frames_l
            index = 0
            count = 0
            while(count < frames_l):
                filerename = fileTem%(startID+index)
                # print filerename
                if os.path.isfile(filerename):
                    video_dict[videokey]['frames'].append(filerename)  # should verify
                    count += 1
                index += 1  
        else:
            for i in range(frames_l):
                video_dict[videokey]['frames'].append(filename)
    else:
         video_dict[videokey]['frames'].append(filelistname.split(' ')[1])

    filename = filelistname.split(' ')[2]
    filepre = filename.split('/')[-1]
    fileser = filepre[:-10]
    fileTem = filename[:-9]+'%05d'+filename[-4:]
    startID = int(filename[-9:-4])
    if frames_l>1:
        file_num = list_num[fileser]
        # print file_num
        endID = startID+ frames_l
        if endID > (file_num[0]+file_num[1]):
            startID = file_num[0]+file_num[1] - frames_l
        index = 0
        count = 0
        while(count < frames_l):
            filerename = fileTem%(startID+index)
            if os.path.isfile(filerename):
                video_dict[videokey]['frames_p'].append(filerename)  # should verify
                count += 1
            index += 1  
    else:
         video_dict[videokey]['frames_p'].append(filelistname.split(' ')[2])  # should verify

    video_dict[videokey]['reshape'] = (height, width)
    video_dict[videokey]['crop'] = (height, width)

    video_dict[videokey]['label'] = []
    video_dict[videokey]['label'].append(int(filelistname.split(' ')[3]))

    return video_dict


class listprocessor(object):
    def __init__(self, frames_l,balance,height,width,list_num):
        self.frames_l = frames_l
        self.balance = balance
        self.height = height
        self.width = width
        self.list_num = list_num

    def __call__(self, filename):
        return processList(filename, self.frames_l, self.balance , self.height, self.width, self.list_num)

def readlistFromFile(video_list,num_list,frames_l,balance,height,width):

    f = open(video_list, 'r')
    f_lines = f.readlines()
    # print f_lines
    f.close()
    f = open(num_list, 'r')
    f_list = f.readlines()
    # print f_lines
    f.close()

    num_dict = {}
    for line in f_list:
        key = '%s_person%03d'%(line.split(' ')[0],int(line.split(' ')[1]))
        # print key
        num_dict[key] = [int(line.split(' ')[2]),int(line.split(' ')[3])]
    # print num_dict
    pool_size = 24
    pool = Pool(processes=pool_size)
    processor = listprocessor(frames_l,balance,height,width,num_dict)
    
    video_dict = {}
    video_dicList = []
    new_lines = []
    video_order = []
    c = 0
    total = len(f_lines)
    point = total/100
    for ix, line in enumerate(f_lines):
        newline = str(ix)+' '+line 
        new_lines.append(newline)
        if ix%point == 0 and ix > 0:
            # print len(new_lines)
            video_dict_buffer = pool.map(processor,new_lines)
            video_dicList.extend(video_dict_buffer)
            
            new_lines = []
            # print ix
            progresslog('data list is loading',ix/point)
    video_dict_buffer = pool.map(processor,new_lines)
    video_dicList.extend(video_dict_buffer)
    # video_order.extend(video_order)
    pool.close()
    pool.join()
    for entity in video_dicList:
        key = entity.keys()[0] 
        video_dict[key] = entity[entity.keys()[0]]
    # video_dict = video_dicList
    video_order = video_dict.keys()
    # print video_order
    print 'video_list:%s'%(video_list)
    print 'list example:'
    print random.choice(video_dict)
    return [video_dict,video_order]
def readlistFromList(Probes,Gallerys,frames_l,balance,height,width):
    probes_dict = Probes[0]
    key_probes = Probes[1]
    gallerys_dict = Gallerys[0]
    key_gallerys = Probes[1]
    video_dict = {}
    video_order = []
    for key in key_probes:
        for keys in key_gallerys:
            videokey = key + keys
            video_dict[videokey] = {}
            video_dict[videokey]['frames'] = []
            video_dict[videokey]['frames_p'] = []

            frame_list = probes_dict[key]
            if len(frame_list)<frames_l :
                video_dict[videokey]['frames'].extend(frame_list*frames_l)
            else:
                video_dict[videokey]['frames'].extend(frame_list)
            framep_list = gallerys_dict[key]
            video_dict[videokey]['frames_p'].extend(framep_list)
            
            video_dict[videokey]['reshape'] = (height, width)
            video_dict[videokey]['crop'] = (height, width)

            video_dict[videokey]['label'] = []

            video_order.append(videokey)
    return [video_dict,video_order]






class videoReadInput(videoRead):
  def initialize(self,phase,flow,batch_size,frames_l,channels,height,width,path_root,Videolist,Numlist,multylabel,affine,balance,Probes = [],Gallerys=[]):
    self.train_or_test = phase #'train'/'test'
    self.flow = flow
    self.buffer_size = batch_size  #num videos processed per batch
    self.frames = frames_l   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.idx = 0
    self.channels = channels
    self.height = height
    self.width = width
    self.path_to_images = path_root  # the pre path to the datasets
    if self.train_or_test == 'train':
        video_dict = readlistFromFile(Videolist,self.frames,balance,self.height,self.width)
    else:
        video_dict = readlistFromList(Probes,Gallerys,self.frames,balance,self.height,self.width)
    self.video_dict = video_dict[0]
    self.video_order = video_dict[1]
    self.num_videos = len(self.video_dict)
    self.multylabel = multylabel #True/False
    self.affine = affine  #True/False

