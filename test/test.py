# -*- coding: utf-8 -*-
import os
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
import sys
sys.path.insert(0, ROOT_DIR)
from dataloader import *
from model_args import *

if __name__ == '__main__':
   model_args = model_argparser()
   provide_data=[('data',(model_args.batch_size,model_args.frame_per_video,3,model_args.height,model_args.width)),]
   provide_label= [('label',(model_args.batch_size,)),]
   train_iter = VideoDataIter(model_args,rec_dir=model_args.train_rec_dir,provide_data=provide_data,provide_label=provide_label) 
   for batch in train_iter:
       print batch
