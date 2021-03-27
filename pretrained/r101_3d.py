# load libs
import sys
sys.path.append('/data/users/bressekk/work/faimed3d')

from fastai.basics import *
from faimed3d.all import *
from fastai.callback.all import *

# global variables
MODEL_DIR = '/data/users/bressekk/work/pretrain_3d/'
DATA_DIR = '/data/users/bressekk/scratch/UCF-101'


# get data
dls = ImageDataLoaders3D.from_folder(DATA_DIR, valid_pct = 0.1, 
                                     item_tfms = Resize3D((20, 112, 112)), 
                                     bs = 32)


# create learner
learn = Learner(dls, 
                resnet101_3d(num_classes = dls.c), 
                cbs = [SaveModelCallback(monitor = 'accuracy', 
                                          fname = 'resnet101_3d_112', 
                                       with_opt = True), 
                      MixUp(), 
                      EarlyStoppingCallback(patience = 50)], 
               model_dir = MODEL_DIR, 
               metrics = accuracy)

learn.fit_one_cycle(250, wd = 1e-4)

# again for 224 x 224 px

# get data
dls = ImageDataLoaders3D.from_folder(DATA_DIR, valid_pct = 0.1, 
                                     item_tfms = Resize3D((20, 224, 224)), 
                                     bs = 16)


# create learner
learn = Learner(dls, 
                resnet101_3d(num_classes = dls.c), 
                cbs = [SaveModelCallback(monitor = 'accuracy', 
                                          fname = 'resnet101_3d_224', 
                                       with_opt = True), 
                      MixUp(), 
                      EarlyStoppingCallback(patience = 50)], 
               model_dir = MODEL_DIR, 
               metrics = accuracy)


learn = learn.load('resnet101_3d_112')
learn.fit_one_cycle(250, wd = 1e-4)
