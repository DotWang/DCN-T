import os
import torch
import scipy.io as scio
import numpy as np
from PIL import Image
from torch.utils import data
from utils.path_utils import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from glob import glob

class TrainDataset(data.Dataset):
    def __init__(self, args, img, target, split_files=None):
        self.args=args
        self.ids=split_files
        self.img = img
        self.target = target
        self.mean=(0.485, 0.456, 0.406)
        self.std=(0.229, 0.224, 0.225)
        #print('Creating dataset with {} examples'.format(len(self.ids)))
        self.transforms = transforms.Compose([
             #tr.RandomScaleCrop(self.args.crop_size),
             #tr.Normalize(mean=self.mean, std=self.std),
             tr.ToTensor()])

    def _class_to_trainid(self, label):

        label_copy = label.copy()

        label_copy -= 1

        label_copy[label_copy < 0] = 255

        return label_copy

    def __len__(self):
        return 1
    def __getitem__(self, i):
        #img_path = self.ids[i]

        #_img = Image.open(img_path)

        _img = self.img

        _target = self.target

        _target = self._class_to_trainid(_target)

        _target = Image.fromarray(_target)

        sample = {'image': _img, 'label': _target}

        sample= self.transforms(sample)

        return sample


class ValDataset(data.Dataset):

    def __init__(self, args, img,  target, split_files=None):
        self.args=args
        self.ids=split_files
        self.img = img
        self.target = target
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        #print('Creating dataset with {} examples'.format(len(self.ids)))
        self.transforms = transforms.Compose([
            #tr.CenterCrop(self.args.crop_size),
            #tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor()])

    def _class_to_trainid(self, label):

        label_copy = label.copy()

        label_copy -= 1

        label_copy[label_copy < 0] = 255

        return label_copy

    def __len__(self):
        return 1
    def __getitem__(self, i):
        # img_path = self.ids[i]
        #
        # _img = Image.open(img_path)

        _img = self.img

        _img = _img[:100,:100,:]

        _target = self.target

        _target = _target[:100,:100]

        _target = self._class_to_trainid(_target)

        _target = Image.fromarray(_target)

        sample = {'image': _img, 'label': _target}

        sample = self.transforms(sample)
        return sample

class TesDataset(data.Dataset):

    def __init__(self, args, img, target,split_files=None):
        self.args=args
        self.ids=split_files
        self.img = img
        self.target = target
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        #print('Creating dataset with {} examples'.format(len(self.ids)))
        self.transforms = transforms.Compose([
            #tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor()])

    def _class_to_trainid(self, label):
        label_copy = label.copy()

        label_copy -= 1

        label_copy[label_copy < 0] = 255

        return label_copy

    def __len__(self):
        return 1

    def __getitem__(self, i):
        # img_path = self.ids[i]
        #
        # _img = Image.open(img_path)

        _img = self.img

        _target = self.target

        _target = self._class_to_trainid(_target)

        _target = Image.fromarray(_target)

        sample = {'image': _img, 'label': _target}

        sample = self.transforms(sample)

        return sample

def make_data_loader(args):
    IMG_SUFFIX = 'png'

    glob_path = os.path.join(Path.db_root_dir(args.dataset)['trnval'], '*.%s' % (IMG_SUFFIX))

    trn_file = glob(glob_path)  # extract all the .png to a list

    strlist = str(args.dataset).split('_')
    if 'LongKou' in strlist[1]:
        prefix = 'LKt'
    elif 'HanChuan' in strlist[1]:
        prefix = 'T'
    elif 'HongHu' in strlist[1]:
        prefix = 'HHCYt'
    else:
        raise NotImplementedError

    img = scio.loadmat('/project/luoyong_01/DW/Dataset/whuhi/Matlab_data_format/Matlab_data_format/WHU-Hi-'+strlist[1]+'/WHU_Hi_'+strlist[1]+'.mat')['WHU_Hi_'+strlist[1]]

    #pre norm

    img = -1 + 2* ((img - np.min(img)) / (np.max(img) - np.min(img)))

    target = scio.loadmat('/project/luoyong_01/DW/Dataset/whuhi/Matlab_data_format/Matlab_data_format/WHU-Hi-'+strlist[1]+'/Training samples and test samples/Train'+strlist[2]+'.mat')[prefix+'rain'+strlist[2]]

    train_set = TrainDataset(args, img, target)

    ix = int(len(trn_file) * 0.95)

    target = scio.loadmat(
        '/project/luoyong_01/DW/Dataset/whuhi/Matlab_data_format/Matlab_data_format/WHU-Hi-' + strlist[
            1] + '/Training samples and test samples/Test' + strlist[2] + '.mat')[prefix + 'est' + strlist[2]]

    val_set = ValDataset(args, img, target)

    if args.distributed=='True':
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                      num_replicas=args.world_size,rank=args.rank)#分布式采样器
    else:
        train_sampler = None

    if args.distributed=='True':
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set,
                                                                      num_replicas=args.world_size,
                                                                      rank=args.rank)  # 分布式采样器
    else:
        val_sampler = None

    if args.distributed == 'True':
        args.batch_size = int(args.batch_size / args.world_size)#将一个节点的BS按GPU平分
        args.test_batch_size = int(args.test_batch_size / args.world_size)
        args.workers = int(args.workers / args.world_size)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, sampler=val_sampler, num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader

