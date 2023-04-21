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
    def __init__(self, args, target, split_files=None):
        self.args=args
        self.ids=split_files
        self.target = target
        self.mean=(0.485, 0.456, 0.406)
        self.std=(0.229, 0.224, 0.225)
        print('Creating dataset with {} examples'.format(len(self.ids)))
        self.transforms = transforms.Compose([
             #tr.RandomCrop(self.args.crop_size),
             #tr.RandomHorizontalFlip(),
             tr.Normalize(mean=self.mean, std=self.std),
             tr.ToTensor()])

    def _class_to_trainid(self, label):

        label_copy = label.copy()

        label_copy -= 1

        label_copy[label_copy < 0] = 255

        return label_copy

    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        img_path = self.ids[i]

        _img = Image.open(img_path)

        #_img = np.load(img_path)

        _target = self.target

        _target = self._class_to_trainid(_target)

        _target = Image.fromarray(_target)

        sample = {'image': _img, 'label': _target}

        sample= self.transforms(sample)

        return sample


class ValDataset(data.Dataset):

    def __init__(self, args, target, split_files=None):
        self.args=args
        self.ids=split_files
        self.target = target
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        print('Creating dataset with {} examples'.format(len(self.ids)))
        self.transforms = transforms.Compose([
            #tr.CenterCrop(self.args.crop_size),
            tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor()])

    def _class_to_trainid(self, label):

        label_copy = label.copy()

        label_copy -= 1

        label_copy[label_copy < 0] = 255

        return label_copy

    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        img_path = self.ids[i]

        _img = Image.open(img_path)

        #_img = np.load(img_path)

        _target = self.target

        _target = self._class_to_trainid(_target)

        _target = Image.fromarray(_target)

        sample = {'image': _img, 'label': _target}

        sample = self.transforms(sample)
        return sample

class TesDataset(data.Dataset):

    def __init__(self, args, target,split_files=None):
        self.args=args
        self.ids=split_files
        self.target = target
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        print('Creating dataset with {} examples'.format(len(self.ids)))
        self.transforms = transforms.Compose([
            tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor()])

    def _class_to_trainid(self, label):
        label_copy = label.copy()

        label_copy -= 1

        label_copy[label_copy < 0] = 255

        return label_copy

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_path = self.ids[i]

        _img = Image.open(img_path)

        #_img = np.load(img_path)

        _target = self.target

        _target = self._class_to_trainid(_target)

        _target = Image.fromarray(_target)

        sample = {'image': _img, 'label': _target}

        sample = self.transforms(sample)

        return sample

def make_data_loader(args):
    IMG_SUFFIX = 'png'

    strlist = str(args.dataset).split('_')

    glob_path = os.path.join('../../Dataset/whu_hi/whuhi_image_2percent/' + strlist[1] + '_' + strlist[2] + '/', '*.%s' % (IMG_SUFFIX))
    
    print(glob_path)

    trn_file = glob(glob_path)  # extract all the .png to a list

    if 'LongKou' in strlist[1]:
        prefix = 'LKt'
    elif 'HanChuan' in strlist[1]:
        prefix = 'T'
    elif 'HongHu' in strlist[1]:
        prefix = 'HHCYt'
    else:
        raise NotImplementedError

    target = scio.loadmat('../../Dataset/whu_hi/Matlab_data_format/Matlab_data_format/WHU-Hi-'+strlist[1]+'/Training samples and test samples/Train'+strlist[3]+'.mat')[prefix+'rain'+strlist[3]]

    ix = int(len(trn_file) * 0.95)
    
    train_set = TrainDataset(args, target, trn_file[:ix])

    target = scio.loadmat(
        '../../Dataset/whu_hi/Matlab_data_format/Matlab_data_format/WHU-Hi-' + strlist[
            1] + '/Training samples and test samples/Test' + strlist[3] + '.mat')[prefix + 'est' + strlist[3]]

    val_set = ValDataset(args, target, trn_file[ix:])

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

