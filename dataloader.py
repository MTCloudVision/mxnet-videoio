#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''

import numpy as np
import mxnet as mx
import random
import os

class VideoDataIter(object): 
    """
    pack  VideoDataIter by rec file
    Parameters:
    ---------
    args：
        image augment argument
    rec_dir:str
          rec file path
    provide_data:
                The name and shape of data provided by this iterator.
    provide_label:
                 The name and shape of label provided by this iterator.
    """    
    def __init__(self,args,rec_dir=None,provide_data=None, provide_label=None):
        if args.is_flow:
           self.data_shape = (1,args.width,args.height)
        else:
           self.data_shape = (3,args.width,args.height) 
        self.resize = args.resize
        self.rand_crop = args.rand_crop
        self.rand_resize = args.rand_resize
        self.rand_mirror = args.rand_mirror
        self.mean = args.mean
        self.std = args.std
        self.batch_size = args.batch_size
        self.frame_per_video = args.frame_per_video
        self.provide_data = provide_data
        self.provide_label = provide_label
        self.rec=[os.path.join(rec_dir, fname) for fname in os.listdir(rec_dir)
                    if os.path.isfile(os.path.join(rec_dir, fname))]
        self.formal_iter_data = mx.io.ImageRecordIter( path_imgrec = self.rec[0],
                                                       aug_seg = mx.image.CreateAugmenter(data_shape=self.data_shape,mean=self.mean,std= self.std,rand_mirror=self.rand_mirror),
                                                       data_name   = 'data',
                                                       label_name  = 'softmax_label',
                                                       batch_size  = self.batch_size * self.frame_per_video,
                                                       data_shape  = self.data_shape,
                                                       preprocess_threads  = 4,
                                                       rand_mirror =   self.rand_mirror,)


    def __iter__(self):
        return self
   

    def next(self):
              try:
                  next_data_batch = next(self.formal_iter_data)
                  formal_data= next_data_batch.data[0]
                  new_data = formal_data.reshape((self.batch_size,self.frame_per_video,self.data_shape[0],self.data_shape[1],self.data_shape[2]))
                  new_data=[new_data]
                  formal_label = next_data_batch.label[0]
                  new_label =  [label.asnumpy()[0] for i ,label in enumerate(formal_label) if i%self.frame_per_video ==0]
                  new_label = mx.nd.array(new_label)
                  new_label =[new_label]
                  return mx.io.DataBatch(data=new_data,label=new_label)
              
              except StopIteration:
                   raise StopIteration
                 
    def reset(self):
        index = random.randint(0,len(self.rec)-1)
        self.formal_iter_data = mx.io.ImageRecordIter( path_imgrec= self.rec[index],
                                                  aug_seg = mx.image.CreateAugmenter(data_shape=self.data_shape,mean=self.mean,std= self.std,rand_mirror=self.rand_mirror),
                                                  data_name           = 'data',
                                                  label_name          = 'softmax_label',
                                                  batch_size          = self.batch_size * self.frame_per_video,
                                                  data_shape          = self.data_shape,
                                                  preprocess_threads  = 4,
                                                  rand_mirror         =   self.rand_mirror,)
    

    
               



