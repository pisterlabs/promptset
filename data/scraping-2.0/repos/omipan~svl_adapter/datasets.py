import torch
import numpy as np
from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision
from calendar import monthrange
import math


def get_dataset(args):
    train_transform, test_transform = im_transforms(args)
    if 'return_alt_pos' not in args:
        args['return_alt_pos'] = False
    train_set_ssl = IMAGE_DATASET(args, train_transform, ['train','val'],args['return_alt_pos'], return_single_image=False)
    
    train_set_lin = IMAGE_DATASET(args, test_transform, ['train'],return_alt_pos=False,return_single_image=True)
    val_set_lin = IMAGE_DATASET(args, test_transform, ['val'],return_alt_pos=False,return_single_image=True)
    test_set_lin = IMAGE_DATASET(args, test_transform, ['test'],return_alt_pos=False,return_single_image=True)
    return train_set_ssl,train_set_lin, val_set_lin,test_set_lin, train_set_lin.perc_1_inds, train_set_lin.perc_10_inds
    
     
def im_transforms(args):
   
    if args['model_type']=='clip':
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=args['im_res'], scale=(0.5, 1), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
                
        
        ### following the transformations that are used in the CLIP model from OpenAI (https://github.com/openai/CLIP)
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=args['im_res'], interpolation=torchvision.transforms.InterpolationMode.BICUBIC,max_size=None, antialias=None),
            torchvision.transforms.CenterCrop(size=(args['im_res'], args['im_res'])),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
    
    else:
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(args['im_res'], scale=(0.2, 1.0)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            get_color_distortion(s=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std= [0.229, 0.224, 0.225])
        ])
        
        
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((args['im_res'], args['im_res'])),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225])
        ])
    
    return train_transform, test_transform
    

# color distortion composed by color jittering and color dropping.
# See Section A of SimCLR: https://arxiv.org/abs/2002.05709
def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
    # s is the strength of color distortion
    color_jitter = torchvision.transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = torchvision.transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = torchvision.transforms.RandomGrayscale(p=0.2)
    color_distort = torchvision.transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort





class IMAGE_DATASET(torch.utils.data.Dataset):

    def __init__(self, args, transform, split_names,return_alt_pos,return_single_image, cache_images=False): 
        da = pd.read_csv(args['metadata'],index_col=0)
        da['im_id'] = np.arange(da.shape[0])

        # get the class labels before some of the data is excluded
        un_targets, targets = np.unique(da['category_id'].values, return_inverse=True) 
        da['category_id_un'] = targets
        self.num_classes = un_targets.shape[0]
        self.context_size = 0 
        
        if (args['model_type']=='ssl') and ('context' in args['pos_type']):
            # load the context data
            context_dict = load_context(da)
            # only use the relevant split's data
            da = da[da['img_set'].isin(split_names)]
            
            inds_to_keep = da['im_id'].values
   
            self.context = torch.tensor(context_dict['con_standard'][inds_to_keep, :])
            self.location_id = context_dict['location_ids'][inds_to_keep]
            self.hour = context_dict['hour'][inds_to_keep]
            self.context_size = self.context.shape[1]
        
            self.context_dict = context_dict
            for kk in ['con_time', 'con_time_scaled', 'con_bbox']:
                self.context_dict[kk] = torch.tensor(self.context_dict[kk][inds_to_keep, :])
        else:
            # only use the relevant split's data
            da = da[da['img_set'].isin(split_names)]
        
        ## return context  used only during ssl_pretraining.py if the context positives approach is picked instead of vanila ssl.
        if 'return_context' not in args:
            self.return_context = False
        else:
            self.return_context = args['return_context']
        

        self.return_alt_pos = return_alt_pos
        self.return_single_image = return_single_image    
        
        self.transform = transform
        self.data_root = args['data_dir']

    
        self.targets = da['category_id_un'].values  # keep as np array
        self.targets_orig = da['category_id'].values  # keep as np array
        self.im_paths = da['img_path'].values.tolist()
        self.alt_paths = [im for im in self.im_paths]  # just initialize as a deep copy
        self.num_examples = len(self.im_paths)


        # random subset for 1,10% subsets (they could be used for SSL evaluation)
        rnd = np.random.RandomState(args['seed'])
        self.perc_1_inds  = rnd.choice(self.num_examples, int(self.num_examples*0.01), replace=False).tolist()        
        self.perc_10_inds = rnd.choice(self.num_examples, int(self.num_examples*0.1), replace=False).tolist()

                
        # cache the image data in RAM, this will use a lot of memory and only makes sense for smallish datasets
        self.cache_images = cache_images
        self.im_cache = {}
        if self.cache_images:
            print('caching images ...')
            for pp in self.im_paths:
                self.im_cache[pp] = loader(self.data_root + pp)                  
                

            print('caching images done\n')
                    
        
    def __len__(self):
        return len(self.im_paths)
        
    
    def get_image(self, root_dir, im_path):  
        if self.cache_images and im_path in self.im_cache:
            return self.im_cache[im_path].copy()
        else:
            return loader(root_dir+im_path)
    
    
    def update_alternative_positives(self, inds):
        for ii, new_ind in enumerate(inds):
            self.alt_paths[ii] = self.im_paths[new_ind]
        
    def __getitem__(self, idx):    
        op = {}
        op['target'] = self.targets[idx] 
        op['target_orig'] = self.targets_orig[idx] 
        op['id'] = idx

        if self.return_context:
            op['location_id'] = self.location_id[idx] 
            op['hour'] = self.hour[idx]
            op['con'] = self.context[idx, :]
        
            
        img1_path = self.im_paths[idx]
        img1 = self.get_image(self.data_root, img1_path)                             
        if self.return_single_image:            
            op['im'] = self.transform(img1)
                
        else:   
            if self.return_alt_pos: 
                # the alt_paths list will be populated periodically (e.g. every epoch) for each image 
                op['im_t1'] = self.transform([img1])[0]
                img2_path = self.alt_paths[idx]
                img2 = self.get_image(self.data_root, img2_path)
                op['im_t2'] = self.transform([img2])[0]
            else:
                # self-augmentation
                op['im_t1'] = self.transform(img1)   ## should i have a self.transform([img1])[0] instead of self.transform(img1) or it depends ?!
                op['im_t2'] = self.transform(img1)   ## should i have a self.transform([img1])[0] instead of self.transform(img1) or it depends ?!

        
        return op


def loader(im_path_full):
    with open(im_path_full, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')     
        
        

def cos_sin_encode(x): 
    #assume betwen 0 and 1
    op = np.zeros((x.shape[0], 2), dtype=np.float32)
    op[:, 0] = np.sin(math.pi*((2*x)-1))
    op[:, 1] = np.cos(math.pi*((2*x)-1))
    op = (op+1)/2.0
    return op
    
def load_context(da, return_box_info=False, return_loc_info=False):

    width = da['width'].values / da['img_width'].values
    height = da['height'].values / da['img_height'].values
    x1 = (da['x1'].values / da['img_width'].values) + (width/2)
    y1 = (da['y1'].values / da['img_height'].values) + (height/2)
    area = (da['width'].values*da['height'].values) / (da['img_width'].values*da['img_height'].values)  
    un_locs, loc_inds = np.unique(da['location'].values, return_inverse=True)
    loc = np.zeros((loc_inds.shape[0], un_locs.shape[0]))
    loc[np.arange(loc.shape[0]), loc_inds] = 1.0  
    more_than_one = (da['boxes_per_img_id'].values>1).astype(np.float32)

    # get count of number of days per year - choose a leap year
    num_days = np.cumsum([0] + [monthrange(2020, ii)[1] for ii in range(1, 13)])
    
    # assuming 24 hour time 
    tm1d = np.zeros(da.shape[0])
    hr1d = pd.to_datetime(da['datetime']).dt.hour.values.astype(np.float32)
    min1d = pd.to_datetime(da['datetime']).dt.minute.values.astype(np.float32)
    sec1d = pd.to_datetime(da['datetime']).dt.second.values.astype(np.float32)
    year = pd.to_datetime(da['datetime']).dt.year.values.astype(np.float32)
    month = pd.to_datetime(da['datetime']).dt.month.values - 1
    day1d = pd.to_datetime(da['datetime']).dt.day.values.astype(np.float32) - 1
    for ii in range(day1d.shape[0]):
        day1d[ii] = num_days[month[ii]] + day1d[ii]
    
    if np.unique(year).shape[0] == 1:
        year = np.zeros(da.shape[0])
    else:
        year -= year.min()
        year /= year.max()
        
    day1d /= 366    
    hr1d /= 23.0
    min1d /= 59.0
    sec1d /= 59.0
    
    day = cos_sin_encode(day1d)
    hr = cos_sin_encode(hr1d)
    mi = cos_sin_encode(min1d)
    sec = cos_sin_encode(sec1d)
    
    year = year[..., np.newaxis]
    x1 = x1[..., np.newaxis]
    y1 = y1[..., np.newaxis]
    width = width[..., np.newaxis]
    height = height[..., np.newaxis]
    area = area[..., np.newaxis]   
    more_than_one = more_than_one[..., np.newaxis]
                         
    op = {}
    op['con_time'] = np.hstack((year, day, hr, mi, sec)).astype(np.float32)
    op['con_time_scaled'] = np.hstack((year*100.0, day*10.0, hr*1.0, mi*0.1, sec*0.01)).astype(np.float32)
    op['con_bbox'] = np.hstack((x1, y1, width, height, more_than_one)).astype(np.float32)
    op['con_loc_onehot'] = loc.astype(np.float32)
    op['con_standard'] = np.hstack((op['con_time'], op['con_loc_onehot']))
    op['location_ids'] = loc_inds
    op['hour'] = (hr1d*23).astype(np.int)
    
    return op


class FeatureDataset(Dataset):
    def __init__(self, data,labels):
        self.data = data
        self.label = labels
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, ind):
        x = self.data[ind]
        y = self.label[ind]
        return x, y
