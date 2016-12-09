import random

from base_input import videoRead
from ProgressBar import *
# def progresslog(s,i):
#     sys.stdout.write(s+':{0:3}/{1:3}:'.format(i, 100)+'#'*i+'->'+'\r')
def format_time(num_time):
    seconds=num_time%60
    minuites=num_time/60%60
    hours=num_time/3600%24
    days=num_time/3600/24
    fm_time={'sec':seconds,'min':minuites,'hours':hours,'days':days}
    return fm_time
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
        filepre = filename[:-7]+'%03d'+filename[-4:]
        person_idp = int(filename.split('/')[-1][:-7])
        excluId = int(filename[-7:-4])
        if frames_l>1:
            for pid in range(4):
                if (pid+1) != excluId:
                    video_dict[videokey]['frames_p'].append(filepre%(pid+1))  # should verify
        else:
             video_dict[videokey]['frames_p'].append(line.split(' ')[1])  # should verify

        video_dict[videokey]['reshape'] = (height, width)
        video_dict[videokey]['crop'] = (height, width)

        video_dict[videokey]['label'] = []
        if multylabel==True:
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
def readlistFromList(Probes,Gallerys,frames_l,balance,height,width,multylabel):
    probes_dict = Probes[0]
    key_probes = Probes[1]
    gallerys_dict = Gallerys[0]
    key_gallerys = Gallerys[1]
    video_dict = {}
    video_order = []
    c = 0
    for key in key_probes:
        for keys in key_gallerys:
            videokey = c
            video_dict[videokey] = {}
            video_dict[videokey]['frames'] = []
            video_dict[videokey]['frames_p'] = []

            frame_list = probes_dict[key]
            if len(frame_list)<frames_l :
                video_dict[videokey]['frames'].extend(frame_list*frames_l)
            else:
                video_dict[videokey]['frames'].extend(frame_list)
            framep_list = gallerys_dict[keys]
            video_dict[videokey]['frames_p'].extend(framep_list)
            
            video_dict[videokey]['reshape'] = (height, width)
            video_dict[videokey]['crop'] = (height, width)

            video_dict[videokey]['label'] = [-1]

            video_order.append(videokey)
            c+=1
    #print video_dict[videokey]
    #print video_order
    #print video_dict
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
    self.multylabel = multylabel #True/False
    self.affine = affine  #True/False
    if self.train_or_test == 'train':
        video_dict = readlistFromFile(Videolist,self.frames,balance,self.height,self.width,self.multylabel)
    else:
        video_dict = readlistFromList(Probes,Gallerys,self.frames,balance,self.height,self.width,self.multylabel)
    self.video_dict = video_dict[0]
    self.video_order = video_dict[1]
    #self.video_order[:10]
    self.num_videos = len(self.video_dict)

