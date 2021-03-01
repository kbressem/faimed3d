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
                                     item_tfms = Resize3D((20, 260, 260)), 
                                     bs = 16)


# create learner
learn = Learner(dls, 
                efficientnet_b2(n_classes = dls.c), 
                cbs = [SaveModelCallback(monitor = 'accuracy', 
                                          fname = 'efficientnet_b2', 
                                       with_opt = True), 
                      MixUp(), 
                      EarlyStoppingCallback(patience = 50)], 
               model_dir = MODEL_DIR, 
               metrics = accuracy)

learn.fit_one_cycle(250, wd = 1e-4)

