#!/usr/bin/env python

# Data layer for video.  Change flow_frames and RGB_frames to be the path to the flow and RGB frames.

import sys

sys.path.append('caffe/python') #should verify
import cv2
import caffe
import numpy as np

import random

from multiprocessing import Pool
from threading import Thread


#video_dict ={'key':'frames'=[],'fames_p'=[],'reshpe'=(h,w),'crop'=(h,w),'label'=[]}
def transform(I,rows,cols,pts1,pts2):
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(I,M,(cols,rows))
    return dst

def format_time(num_time):
    seconds=num_time%60
    minuites=num_time/60%60
    hours=num_time/3600%24
    days=num_time/3600/24
    fm_time={'sec':seconds,'min':minuites,'hours':hours,'days':days}
    return fm_time


def processImageCrop(im_info, transformer, flow, affine):
    im_path = im_info[0]
    im_crop = im_info[1]
    im_reshape = im_info[2]
    im_affine = im_info[3]
    im_flip = im_info[4]

    data_in = caffe.io.load_image(im_path)
    if len(im_affine)==3 :
        M = im_reshape[0]
        N = im_reshape[1]
        
        pts1 = np.float32([[0,0],[N,0],[0,M]])
        dx = im_affine[0]
        dy = im_affine[1]
        ds = im_affine[2]
        ds_x = (N-(1+ds)*N)/2
        ds_y = (M-(1+ds)*M)/2
        if random.uniform(0,1) > 0.3:
            pts2 = np.float32([[dx+ds_x,dy+ds_y],[N+dx-ds_x,dy+ds_y],[dx+ds_x,M+dy-ds_y]])
        else:
            pts2 = np.float32([[N+dx-ds_x,dy+ds_y],[dx+ds_x,dy+ds_y],[N+dx-ds_x,M+dy-ds_y]])
            
        rows,cols,c = data_in.shape
        data_in = transform(data_in,rows,cols,pts1,pts2)
    
    if (data_in.shape[0] < im_reshape[0]) | (data_in.shape[1] < im_reshape[1]):
        data_in = caffe.io.resize_image(data_in, im_reshape)
    if im_flip:
        # data_in = caffe.io.flip_image(data_in, 1, flow)
        data_in = data_in[im_crop[0]:im_crop[2], im_crop[1]:im_crop[3], :]
    processed_image = transformer.preprocess('data_in', data_in)
    return processed_image


class ImageProcessorCrop(object):
    def __init__(self, transformer, flow, affine):
        self.transformer = transformer
        self.flow = flow
        self.affine = affine

    def __call__(self, im_info):
        return processImageCrop(im_info, self.transformer, self.flow, self.affine)


class sequenceGeneratorVideo(object):
    def __init__(self, buffer_size, clip_length, num_videos, video_dict, video_order, affine):
        self.buffer_size = buffer_size
        self.clip_length = clip_length
        self.N = self.buffer_size * self.clip_length
        self.num_videos = num_videos
        self.video_dict = video_dict
        self.video_order = video_order
        self.idx = 0
        self.affine = affine
    def __call__(self):
        label_r = []
        im_paths = []
        im_paths_p = []
        im_crop = []
        im_reshape = []
        im_flip = []
        im_affine = []
        im_affine_p = []

        if self.idx + self.buffer_size >= self.num_videos:
            idx_list = range(self.idx, self.num_videos)
            idx_list.extend(range(0, self.buffer_size - (self.num_videos - self.idx)))
        else:
            idx_list = range(self.idx, self.idx + self.buffer_size)

        for i in idx_list:
            key = self.video_order[i]
            #print key
            label = self.video_dict[key]['label']
            video_reshape = self.video_dict[key]['reshape']
            video_crop = self.video_dict[key]['crop']
            label_r.extend([label] * self.clip_length)

            im_reshape.extend([(video_reshape)] * self.clip_length)
            r0 = int(random.random() * (video_reshape[0] - video_crop[0]))
            r1 = int(random.random() * (video_reshape[1] - video_crop[1]))
            im_crop.extend([(r0, r1, r0 + video_crop[0], r1 + video_crop[1])] * self.clip_length)
            f = random.randint(0, 1)
            im_flip.extend([f] * self.clip_length)
            
            if self.affine == True:
                ratio = 0.05
                ratio_s = 0.05
                M = video_reshape[0]
                N = video_reshape[1]
                dx = random.uniform(N*(-ratio), N*(ratio))
                dy = random.uniform(M*(-ratio), M*(ratio))
                ds = random.uniform(-ratio_s,ratio_s)
                im_affine.extend([(dx,dy,ds)] * self.clip_length)
                dx_p = random.uniform(N*(-ratio), N*(ratio))
                dy_p = random.uniform(M*(-ratio), M*(ratio))
                ds_p = random.uniform(-ratio_s,ratio_s)
                im_affine_p.extend([(dx_p,dy_p,ds_p)] * self.clip_length)
            else:
                im_affine.extend([(-1,-1)]*self.clip_length)
                im_affine_p.extend([(-1,-1)]*self.clip_length)
            frames = []
            frames.extend(self.video_dict[key]['frames'])
            #print frames

            im_paths.extend(frames)
            frames_p = []
            frames_p.extend(self.video_dict[key]['frames_p'])
            im_paths_p.extend(frames_p)

        im_info = zip(im_paths, im_crop, im_reshape, im_affine, im_flip)
        im_info_p = zip(im_paths_p, im_crop, im_reshape, im_affine, im_flip)

        self.idx += self.buffer_size
        if self.idx >= self.num_videos:
            self.idx = self.idx - self.num_videos

        return label_r, im_info , im_info_p


def advance_batch(result, sequence_generator, image_processor, pool, clip_len):
    label_r, im_info ,im_info_p = sequence_generator()
    #print im_info[:100]
    tmp = image_processor(im_info[0])
    result['data'] = pool.map(image_processor, im_info)
    #print len(result['data'])
    result['data_p'] = pool.map(image_processor, im_info_p)
    result['label'] = label_r

    cm = np.ones(len(label_r))
    cm[0::clip_len] = 0
    result['clip_markers'] = cm


class BatchAdvancer():
    def __init__(self, result, sequence_generator, image_processor, pool, clip_len):
        self.result = result
        self.sequence_generator = sequence_generator
        self.image_processor = image_processor
        self.pool = pool
        self.clip_len = clip_len
    def __call__(self):
        return advance_batch(self.result, self.sequence_generator, self.image_processor, self.pool, self.clip_len)


class videoRead():
    def initialize(self, phase, flow, batch_size, frames_l, channels, height, width, path_root, Videolist, Numlist, multylabel,
                   affine,balance,Probes = {},Gallerys={}):
        self.train_or_test = phase  # 'train'/'test'
        self.flow = flow
        self.buffer_size = batch_size  # num videos processed per batch
        self.frames = frames_l  # length of processed clip
        self.N = self.buffer_size * self.frames
        self.idx = 0
        self.channels = channels
        self.height = height
        self.width = width
        self.path_to_images = path_root  # the pre path to the datasets
        self.video_dict = {}
        self.video_order = []
        self.num_videos = len(self.video_dict)
        self.multylabel = multylabel  # True/False
        self.affine = affine  # True/False

    def setup(self):
        random.seed(10)
        #print self.video_order[:10]
        # set up data transformer
        shape = (self.N, self.channels, self.height, self.width)
        
        self.transformer = caffe.io.Transformer({'data_in': shape})
        self.transformer.set_raw_scale('data_in', 255)
        if self.flow:
            image_mean = [128, 128, 128]
            # self.transformer.set_is_flow('data_in', True)
        else:
            image_mean = [104, 117, 123]
            # self.transformer.set_is_flow('data_in', False)
        channel_mean = np.zeros((3, self.height, self.width))
        for channel_index, mean_val in enumerate(image_mean):
            channel_mean[channel_index, ...] = mean_val
        self.transformer.set_mean('data_in', channel_mean)
        self.transformer.set_channel_swap('data_in', (2, 1, 0))
        self.transformer.set_transpose('data_in', (2, 0, 1))

        self.thread_result = {}
        self.thread = None
        pool_size = 6

        self.image_processor = ImageProcessorCrop(self.transformer, self.flow, self.affine)
        self.sequence_generator = sequenceGeneratorVideo(self.buffer_size, self.frames, self.num_videos,
                                                         self.video_dict, self.video_order, self.affine)

        self.pool = Pool(processes=pool_size)
        self.batch_advancer = BatchAdvancer(self.thread_result, self.sequence_generator, self.image_processor,
                                            self.pool, self.buffer_size)
        self.dispatch_worker()
        if self.frames>1:
            self.top_names = ['data', 'data_p', 'label', 'clip_markers']
        else:
            self.top_names = ['data', 'data_p', 'label']
        if self.multylabel:
            self.top_names.extend(['label_ID','label_pID'])
        if self.train_or_test == 'test':
            self.top_names.remove('label')
        print 'Outputs:', self.top_names
        
        self.join_worker()

    def reshape(self, net):
        pass

    def forward(self, net):

        if self.thread is not None:
            self.join_worker()

        # rearrange the data: The LSTM takes inputs as [video0_frame0, video1_frame0,...] but the data is currently arranged as [video0_frame0, video0_frame1, ...]
        new_result_data = [None] * len(self.thread_result['data'])
        new_result_data_p = [None] * len(self.thread_result['data_p'])
        new_result_label = [None] * len(self.thread_result['label'])
        new_result_cm = [None] * len(self.thread_result['clip_markers'])
        for i in range(self.frames):
            for ii in range(self.buffer_size):
                old_idx = ii * self.frames + i
                new_idx = i * self.buffer_size + ii
                new_result_data[new_idx] = self.thread_result['data'][old_idx]
                new_result_data_p[new_idx] = self.thread_result['data_p'][old_idx]
                new_result_label[new_idx] = self.thread_result['label'][old_idx]
                new_result_cm[new_idx] = self.thread_result['clip_markers'][old_idx]
        label_array = np.array(new_result_label)
        

        for top_index, name in zip(range(len(self.top_names)), self.top_names):
            if name == 'data':
                if self.train_or_test=='test':
                    net.blobs[name].reshape(self.N,self.channels,self.height,self.width)
                for i in range(self.N):
                    net.blobs[name].data[i, ...] = new_result_data[i]
            elif name == 'data_p':
                if self.train_or_test=='test':
                    net.blobs[name].reshape(self.N,self.channels,self.height,self.width)
                for i in range(self.N):
                    net.blobs[name].data[i, ...] = new_result_data_p[i]
            elif name == 'label' :
                if self.train_or_test=='test':
                    net.blobs[name].reshape(self.N)
                net.blobs[name].data[...] = label_array[:,0]
            elif name == 'clip_markers':
                if self.train_or_test=='test':
                    net.blobs[name].reshape(self.N)
                net.blobs[name].data[...] = new_result_cm
            elif name == 'label_ID':
                if self.train_or_test=='test':
                    net.blobs[name].reshape(self.N)
                net.blobs[name].data[...] = label_array[:,1]
            elif name == 'label_pID':
                if self.train_or_test=='test':
                    net.blobs[name].reshape(self.N)
                net.blobs[name].data[...] = label_array[:,2]
        #print top[2].data[...] 
        self.dispatch_worker()

    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None

    def backward(self, net):
        pass