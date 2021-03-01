#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append('/data/users/bressekk/work/faimed3d')
sys.path.append('..')


# # Pretrain Models on UCF-101
# 
# 3-CV with UCF-101. Train/Test splits are provided in txt files, so at first the txt files are converted to `pd.DataFrame` for easier use.

# In[ ]:


from faimed3d.all import *
from fastai.callback.all import *
from fastai.distributed import *
import faimed3d


# In[ ]:


BASE_DIR = Path('/data/users/bressekk/scratch/')
if not BASE_DIR.exists(): 
    BASE_DIR = Path('/media/ScaleOut/vahldiek/Vids')
DATA_DIR = BASE_DIR/'UCF-101'

MODEL_DIR = Path('/data/users/bressekk/work/pretrain_3d/')
if not MODEL_DIR.exists(): 
    MODEL_DIR = Path(os.getcwd())

MODELS = ['resnet18_3d', 'resnet50_3d', 'resnet101_3d', 'efficientnet_b0', 'efficientnet_b1']


# In[ ]:


def txt_to_df(fn):
    file_dict = {'video_name':[], 
                 'action_type':[]
            } 

    with open(fn, 'r') as f: 
        for l in f.readlines():
            action_type, file_name = l.split('/')
            file_name = file_name.split(' ')[0]
            file_dict['video_name'].append(DATA_DIR/action_type/file_name.strip()) 
            file_dict['action_type'].append(action_type)

    return pd.DataFrame.from_dict(file_dict)


# In[ ]:


train1 = BASE_DIR/'UCF101TrainTestSplits-RecognitionTask'/'ucfTrainTestlist'/'trainlist01.txt'
train2 = BASE_DIR/'UCF101TrainTestSplits-RecognitionTask'/'ucfTrainTestlist'/'trainlist02.txt'
train3 = BASE_DIR/'UCF101TrainTestSplits-RecognitionTask'/'ucfTrainTestlist'/'trainlist03.txt'

test1 = BASE_DIR/'UCF101TrainTestSplits-RecognitionTask'/'ucfTrainTestlist'/'testlist01.txt'
test2 = BASE_DIR/'UCF101TrainTestSplits-RecognitionTask'/'ucfTrainTestlist'/'testlist02.txt'
test3 = BASE_DIR/'UCF101TrainTestSplits-RecognitionTask'/'ucfTrainTestlist'/'testlist03.txt'


# In[ ]:


def get_learner(dls, model_name, model_dir, image_size, split_idx):
    model_func = getattr(faimed3d.all, model_name)
    learn = Learner(dls, 
                    model_func(num_classes = dls.c), 
                    cbs = [SaveModelCallback(monitor = 'accuracy', 
                                             fname = f'{model_name}_{image_size}px_split_{split_idx}', 
                                             with_opt = True), 
                          MixUp(), 
                          EarlyStoppingCallback(patience = 25), 
                          CSVLogger(fname=model_dir/f'{model_name}_{image_size}px_split_{split_idx}.csv')], 
                   model_dir = model_dir, 
                   metrics = accuracy)
    return learn


# In[ ]:


def train_one_split(model_name, train_source, split_idx):
    
    test_source = Path(str(train_source).replace('train', 'test'))
    
    train = txt_to_df(train_source)
    train['is_valid'] = 0
    test = txt_to_df(test_source)
    test['is_valid'] = 1
    df = pd.concat((train, test))
        
    # start small size
    dls = ImageDataLoaders3D.from_df(df, path = '/', 
                                 bs = 32, 
                                 splitter = ColSplitter('is_valid'),
                                 item_tfms = Resize3D((20, 112, 112)))   
    learn = get_learner(dls, model_name=model_name, model_dir=MODEL_DIR, image_size=112, split_idx=split_idx)
    
    with learn.parallel_ctx(): learn.fit_one_cycle(150, wd = 1e-4)
    
    #prog resizing
    new_size = 240 if model_name == 'efficientnet_b1' else 224
    
    dls = ImageDataLoaders3D.from_df(df, path = '/', 
                                 bs = 16, 
                                 splitter = ColSplitter('is_valid'), 
                                 item_tfms = Resize3D((20, new_size, new_size)))
    learn = get_learner(dls, model_name=model_name, model_dir=MODEL_DIR, image_size=new_size, split_idx=split_idx)
    learn = learn.load(f'{model_name}_{112}px_split_{split_idx}')
    
    with learn.parallel_ctx(): learn.fit_one_cycle(200, wd = 1e-4)


# Run the training

# In[ ]:


for model in MODELS:
    for split, data in zip([1,2,3], [train1, train2, train3]):
        print(f'{model}, split: {split}')
        train_one_split(model, data, split)


# In[ ]:




