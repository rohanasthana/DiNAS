""" NTK and NLR calculation 
Based on code from https://github.com/VITA-Group/TENAS """

import torch
import os.path as osp
import numpy as np
import random, os, sys, json
from pathlib import Path
from collections import namedtuple

import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from copy import deepcopy
from PIL import Image
from functools import reduce
from operator import mul
from src import Settings

lib_dir = (Path(__file__).parent / os.path.join(Settings.FOLDER,'datasets/nasbench201/lib')).resolve() 
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))   
from models import get_cell_based_tiny_net, get_search_spaces 

support_types = ('str', 'int', 'bool', 'float', 'none')


def convert_param(original_lists):
    assert isinstance(original_lists, list), 'The type is not right : {:}'.format(original_lists)
    ctype, value = original_lists[0], original_lists[1]
    assert ctype in support_types, 'Ctype={:}, support={:}'.format(ctype, support_types)
    is_list = isinstance(value, list)
    if not is_list: value = [value]
    outs = []
    for x in value:
        if ctype == 'int':
            x = int(x)
        elif ctype == 'str':
            x = str(x)
        elif ctype == 'bool':
            x = bool(int(x))
        elif ctype == 'float':
            x = float(x)
        elif ctype == 'none':
            if x.lower() != 'none':
                raise ValueError('For the none type, the value must be none instead of {:}'.format(x))
            x = None
        else:
            raise TypeError('Does not know this type : {:}'.format(ctype))
        outs.append(x)
    if not is_list: outs = outs[0]
    return outs

def load_config_dict(path):
    path = str(path)
    assert os.path.exists(path), 'Can not find {:}'.format(path)
    # Reading data back
    with open(path, 'r') as f:
        data = json.load(f)
    content = {k: convert_param(v) for k, v in data.items()}
    return content


def load_config(path, extra, logger=None):
    if hasattr(logger, 'log'):
        logger.log(path)
    content = load_config_dict(path)
    assert extra is None or isinstance(
        extra, dict), 'invalid type of extra : {:}'.format(extra)
    if isinstance(extra, dict):
        content = {**content, **extra}
    Arguments = namedtuple('Configure', ' '.join(content.keys()))
    content = Arguments(**content)
    if hasattr(logger, 'log'):
        logger.log('{:}'.format(content))
    return content

class SearchDataset(data.Dataset):

    def __init__(self, name, data, train_split, valid_split, check=True):
        self.datasetname = name
        if isinstance(data, (list, tuple)): # new type of SearchDataset
            assert len(data) == 2, 'invalid length: {:}'.format( len(data) )
            self.train_data  = data[0]
            self.valid_data  = data[1]
            self.train_split = train_split.copy()
            self.valid_split = valid_split.copy()
            self.mode_str    = 'V2' # new mode
        else:
            self.mode_str    = 'V1' # old mode
            self.data        = data
            self.train_split = train_split.copy()
            self.valid_split = valid_split.copy()
            if check:
                intersection = set(train_split).intersection(set(valid_split))
                assert len(intersection) == 0, 'the splitted train and validation sets should have no intersection'
        self.length      = len(self.train_split)

    def __repr__(self):
        return ('{name}(name={datasetname}, train={tr_L}, valid={val_L}, version={ver})'.format(name=self.__class__.__name__, datasetname=self.datasetname, tr_L=len(self.train_split), val_L=len(self.valid_split), ver=self.mode_str))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index >= 0 and index < self.length, 'invalid index = {:}'.format(index)
        train_index = self.train_split[index]
        valid_index = random.choice( self.valid_split )
        if self.mode_str == 'V1':
            train_image, train_label = self.data[train_index]
            valid_image, valid_label = self.data[valid_index]
        elif self.mode_str == 'V2':
            train_image, train_label = self.train_data[train_index]
            valid_image, valid_label = self.valid_data[valid_index]
        else: raise ValueError('invalid mode : {:}'.format(self.mode_str))
        return train_image, train_label, valid_image, valid_label

Dataset2Class = {'cifar10': 10,
                 'cifar100': 100,
                 'imagenet-1k-s': 1000,
                 'imagenet-1k': 1000,
                 'ImageNet16' : 1000,
                 'ImageNet16-150': 150,
                 'ImageNet16-120': 120,
                 'ImageNet16-200': 200}


class CUTOUT(object):

    def __init__(self, length):
        self.length = length

    def __repr__(self):
        return ('{name}(length={length})'.format(name=self.__class__.__name__, **self.__dict__))

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


imagenet_pca = {
        'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
        'eigvec': np.asarray([
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
        ])
}


class Lighting(object):
    def __init__(self, alphastd,
                  eigval=imagenet_pca['eigval'],
                  eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_datasets(name, root, cutout):

    if name == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std  = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif name == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std  = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif name.startswith('imagenet-1k'):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif name.startswith('ImageNet16'):
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std  = [x / 255 for x in [63.22,  61.26 , 65.09]]
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    # Data Argumentation
    if name == 'cifar10' or name == 'cifar100':
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)]
        if cutout > 0 : lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        xshape = (1, 3, 32, 32)
    elif name.startswith('ImageNet16'):
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2), transforms.ToTensor(), transforms.Normalize(mean, std)]
        if cutout > 0 : lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        xshape = (1, 3, 16, 16)
    elif name == 'tiered':
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(80, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)]
        if cutout > 0 : lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.CenterCrop(80), transforms.ToTensor(), transforms.Normalize(mean, std)])
        xshape = (1, 3, 32, 32)
    elif name.startswith('imagenet-1k'):
        lists = [
                    transforms.Resize((32, 32), interpolation=2),
                    transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)
                ]
        if cutout > 0 : lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        xshape = (1, 3, 32, 32)
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    if name == 'cifar10':
        train_data = dset.CIFAR10 (root, train=True , transform=train_transform, download=True)
        test_data  = dset.CIFAR10 (root, train=False, transform=test_transform , download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == 'cifar100':
        train_data = dset.CIFAR100(root, train=True , transform=train_transform, download=True)
        test_data  = dset.CIFAR100(root, train=False, transform=test_transform , download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name.startswith('imagenet-1k'):
        train_data = dset.ImageFolder(osp.join(root, 'train'), train_transform)
        test_data  = dset.ImageFolder(osp.join(root, 'val'),   test_transform)
    elif name == 'ImageNet16':
        train_data = ImageNet16(root, True , train_transform)
        test_data  = ImageNet16(root, False, test_transform)
        assert len(train_data) == 1281167 and len(test_data) == 50000
    elif name == 'ImageNet16-120':
        train_data = ImageNet16(root, True , train_transform, 120)
        test_data  = ImageNet16(root, False, test_transform , 120)
        assert len(train_data) == 151700 and len(test_data) == 6000
    elif name == 'ImageNet16-150':
        train_data = ImageNet16(root, True , train_transform, 150)
        test_data  = ImageNet16(root, False, test_transform , 150)
        assert len(train_data) == 190272 and len(test_data) == 7500
    elif name == 'ImageNet16-200':
        train_data = ImageNet16(root, True , train_transform, 200)
        test_data  = ImageNet16(root, False, test_transform , 200)
        assert len(train_data) == 254775 and len(test_data) == 10000
    else: raise TypeError("Unknow dataset : {:}".format(name))

    class_num = Dataset2Class[name]
    return train_data, test_data, xshape, class_num


def get_nas_search_loaders(train_data, valid_data, dataset, config_root, batch_size, workers):
    if isinstance(batch_size, (list,tuple)):
        batch, test_batch = batch_size
    else:
        batch, test_batch = batch_size, batch_size
    if dataset == 'cifar10':
        cifar_split = load_config('{:}/cifar-split.txt'.format(config_root), None, None)
        train_split, valid_split = cifar_split.train, cifar_split.valid # search over the proposed training and validation set
        # To split data
        xvalid_data  = deepcopy(train_data)
        if hasattr(xvalid_data, 'transforms'): # to avoid a print issue
            xvalid_data.transforms = valid_data.transform
        xvalid_data.transform  = deepcopy( valid_data.transform )
        search_data   = SearchDataset(dataset, train_data, train_split, valid_split)
        # data loader
        search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, shuffle=True , num_workers=workers, pin_memory=True)
        train_loader  = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True)
        valid_loader  = torch.utils.data.DataLoader(xvalid_data, batch_size=test_batch, sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split), num_workers=workers, pin_memory=True)
    elif dataset == 'cifar100':
        cifar100_test_split = load_config('{:}/cifar100-test-split.txt'.format(config_root), None, None)
        search_train_data = train_data
        search_valid_data = deepcopy(valid_data) ; search_valid_data.transform = train_data.transform
        search_data   = SearchDataset(dataset, [search_train_data,search_valid_data], list(range(len(search_train_data))), cifar100_test_split.xvalid)
        search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, shuffle=True , num_workers=workers, pin_memory=True)
        train_loader  = torch.utils.data.DataLoader(train_data , batch_size=batch, shuffle=True , num_workers=workers, pin_memory=True)
        valid_loader  = torch.utils.data.DataLoader(valid_data , batch_size=test_batch, shuffle=False, num_workers=workers, pin_memory=True)
    elif dataset == 'ImageNet16-120':
        imagenet_test_split = load_config('{:}/imagenet-16-120-test-split.txt'.format(config_root), None, None)
        search_train_data = train_data
        search_valid_data = deepcopy(valid_data) ; search_valid_data.transform = train_data.transform
        search_data   = SearchDataset(dataset, [search_train_data,search_valid_data], list(range(len(search_train_data))), imagenet_test_split.xvalid)
        search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, shuffle=True , num_workers=workers, pin_memory=True)
        train_loader  = torch.utils.data.DataLoader(train_data , batch_size=batch, shuffle=True , num_workers=workers, pin_memory=True)
        valid_loader  = torch.utils.data.DataLoader(valid_data , batch_size=test_batch, sampler=torch.utils.data.sampler.SubsetRandomSampler(imagenet_test_split.xvalid), num_workers=workers, pin_memory=True)
    else:
        raise ValueError('invalid dataset : {:}'.format(dataset))
    return search_loader, train_loader, valid_loader


def get_ntk_n(xloader, networks, recalbn=0, train_mode=False, num_batch=-1):
    device = 'cpu'
    ntks = []
    
    for network in networks:
        #network.to(device)
        if train_mode:
            network.train()
        else:
            network.eval()
    ######
    grads = [[] for _ in range(len(networks))]
    for i, (inputs, targets) in enumerate(xloader):
        if num_batch > 0 and i >= num_batch: break
        inputs = inputs.to(device)
        for net_idx, network in enumerate(networks):
            network.zero_grad()
            inputs_ = inputs.clone().to(device=device, non_blocking=True)
            logit = network(inputs_)
            if isinstance(logit, tuple):
                logit = logit[1]  # 201 networks: return features and logits
            for _idx in range(len(inputs_)):
                logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                grad = []
                for name, W in network.named_parameters():
                    if 'weight' in name and W.grad is not None:
                        grad.append(W.grad.view(-1).detach())
                grads[net_idx].append(torch.cat(grad, -1))
                network.zero_grad()
                torch.cuda.empty_cache()
    ######
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
    conds = []
    for ntk in ntks:
        eigenvalues, _ = torch.symeig(ntk)  # ascending
        conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
    return conds

def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def kaiming_normal_fanout_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)

def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    elif method == 'kaiming_norm_fanout':
        model.apply(kaiming_normal_fanout_init)
    return model

class RandChannel(object):
    # randomly pick channels from input
    def __init__(self, num_channel):
        self.num_channel = num_channel

    def __repr__(self):
        return ('{name}(num_channel={num_channel})'.format(name=self.__class__.__name__, **self.__dict__))

    def __call__(self, img):
        channel = img.size(0)
        channel_choice = sorted(np.random.choice(list(range(channel)), size=self.num_channel, replace=False))
        return torch.index_select(img, 0, torch.Tensor(channel_choice).long())

def get_datasets_lr(name, root, input_size, cutout=-1):
    assert len(input_size) in [3, 4]
    if len(input_size) == 4:
        input_size = input_size[1:]
    assert input_size[1] == input_size[2]

    if name == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std  = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif name == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std  = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif name.startswith('imagenet-1k'):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif name.startswith('ImageNet16'):
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std  = [x / 255 for x in [63.22,  61.26 , 65.09]]
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    # Data Argumentation
    if name == 'cifar10' or name == 'cifar100':
        lists = [transforms.RandomCrop(input_size[1], padding=0), transforms.ToTensor(), transforms.Normalize(mean, std), RandChannel(input_size[0])]
        if cutout > 0 : lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    elif name.startswith('ImageNet16'):
        lists = [transforms.RandomCrop(input_size[1], padding=0), transforms.ToTensor(), transforms.Normalize(mean, std), RandChannel(input_size[0])]
        if cutout > 0 : lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    elif name.startswith('imagenet-1k'):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if name == 'imagenet-1k':
            xlists    = []
            xlists.append(transforms.Resize((32, 32), interpolation=2))
            xlists.append(transforms.RandomCrop(input_size[1], padding=0))
        elif name == 'imagenet-1k-s':
            xlists = [transforms.RandomResizedCrop(32, scale=(0.2, 1.0))]
            xlists = []
        else: raise ValueError('invalid name : {:}'.format(name))
        xlists.append(transforms.ToTensor())
        xlists.append(normalize)
        xlists.append(RandChannel(input_size[0]))
        train_transform = transforms.Compose(xlists)
        test_transform = transforms.Compose([transforms.Resize(40), transforms.CenterCrop(32), transforms.ToTensor(), normalize])
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    if name == 'cifar10':
        train_data = dset.CIFAR10 (root, train=True , transform=train_transform, download=True)
        test_data  = dset.CIFAR10 (root, train=False, transform=test_transform , download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == 'cifar100':
        train_data = dset.CIFAR100(root, train=True , transform=train_transform, download=True)
        test_data  = dset.CIFAR100(root, train=False, transform=test_transform , download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name.startswith('imagenet-1k'):
        train_data = dset.ImageFolder(osp.join(root, 'train'), train_transform)
        test_data  = dset.ImageFolder(osp.join(root, 'val'),   test_transform)
    elif name == 'ImageNet16':
        train_data = ImageNet16(root, True , train_transform)
        test_data  = ImageNet16(root, False, test_transform)
        assert len(train_data) == 1281167 and len(test_data) == 50000
    elif name == 'ImageNet16-120':
        train_data = ImageNet16(root, True , train_transform, 120)
        test_data  = ImageNet16(root, False, test_transform , 120)
        assert len(train_data) == 151700 and len(test_data) == 6000
    elif name == 'ImageNet16-150':
        train_data = ImageNet16(root, True , train_transform, 150)
        test_data  = ImageNet16(root, False, test_transform , 150)
        assert len(train_data) == 190272 and len(test_data) == 7500
    elif name == 'ImageNet16-200':
        train_data = ImageNet16(root, True , train_transform, 200)
        test_data  = ImageNet16(root, False, test_transform , 200)
        assert len(train_data) == 254775 and len(test_data) == 10000
    else: raise TypeError("Unknow dataset : {:}".format(name))

    class_num = Dataset2Class[name]
    return train_data, test_data, class_num

class LinearRegionCount(object):
    """Computes and stores the average and current value"""
    def __init__(self, n_samples):
        self.ActPattern = {}
        self.n_LR = -1
        self.n_samples = n_samples
        self.ptr = 0
        self.activations = None

    @torch.no_grad()
    def update2D(self, activations):
        device =  'cpu'
        n_batch = activations.size()[0]
        n_neuron = activations.size()[1]
        self.n_neuron = n_neuron
        if self.activations is None:
            self.activations = torch.zeros(self.n_samples, n_neuron).to(device)
        self.activations[self.ptr:self.ptr+n_batch] = torch.sign(activations)  # after ReLU
        self.ptr += n_batch

    @torch.no_grad()
    def calc_LR(self):
        res = torch.matmul(self.activations, (1-self.activations).T) # each element in res: A * (1 - B)
        res = res + res.T
        res = 1 - torch.sign(res) # a non-zero element now indicate two linear regions are identical
        res = res.sum(1) # for each sample's linear region: how many identical regions from other samples
        res = 1. / res.float() # contribution of each redudant (repeated) linear region
        self.n_LR = res.sum().item() # sum of unique regions (by aggregating contribution of all regions)
        del self.activations, res
        self.activations = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def update1D(self, activationList):
        code_string = ''
        for key, value in activationList.items():
            n_neuron = value.size()[0]
            for i in range(n_neuron):
                if value[i] > 0:
                    code_string += '1'
                else:
                    code_string += '0'
        if code_string not in self.ActPattern:
            self.ActPattern[code_string] = 1

    def getLinearReginCount(self):
        if self.n_LR == -1:
            self.calc_LR()
        return self.n_LR


class Linear_Region_Collector:
    def __init__(self, models=[], input_size=(64, 3, 32, 32), sample_batch=100, dataset='cifar100', data_path=None, seed=0):
        self.models = []
        self.input_size = input_size  # BCHW
        self.sample_batch = sample_batch
        self.input_numel = reduce(mul, self.input_size, 1)
        self.interFeature = []
        self.dataset = dataset
        self.data_path = data_path
        self.seed = seed
        self.reinit(models, input_size, sample_batch, seed)

    def reinit(self, models=None, input_size=None, sample_batch=None, seed=None):
        if models is not None:
            assert isinstance(models, list)
            del self.models
            self.models = models
            for model in self.models:
                self.register_hook(model)
            self.LRCounts = [LinearRegionCount(self.input_size[0]*self.sample_batch) for _ in range(len(models))]
        if input_size is not None or sample_batch is not None:
            if input_size is not None:
                self.input_size = input_size  # BCHW
                self.input_numel = reduce(mul, self.input_size, 1)
            if sample_batch is not None:
                self.sample_batch = sample_batch
            if self.data_path is not None:
                self.train_data, _, class_num = get_datasets_lr(self.dataset, self.data_path, self.input_size, -1)
                self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.input_size[0], num_workers=16, pin_memory=True, drop_last=True, shuffle=True)
                self.loader = iter(self.train_loader)
        if seed is not None and seed != self.seed:
            self.seed = seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def clear(self):
        self.LRCounts = [LinearRegionCount(self.input_size[0]*self.sample_batch) for _ in range(len(self.models))]
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def register_hook(self, model):
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                m.register_forward_hook(hook=self.hook_in_forward)

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.interFeature.append(output.detach())  # for ReLU

    def forward_batch_sample(self):
        for _ in range(self.sample_batch):
            try:
                inputs, targets = self.loader.next()
            except Exception:
                del self.loader
                self.loader = iter(self.train_loader)
                inputs, targets = self.loader.next()
            for model, LRCount in zip(self.models, self.LRCounts):
                self.forward(model, LRCount, inputs)
        return [LRCount.getLinearReginCount() for LRCount in self.LRCounts]

    def forward(self, model, LRCount, input_data):
        self.interFeature = []
        device = 'cpu'
        with torch.no_grad():
            model.forward(input_data.to(device))
            if len(self.interFeature) == 0: return
            feature_data = torch.cat([f.view(input_data.size(0), -1) for f in self.interFeature], 1)
            LRCount.update2D(feature_data)

            
space = 'darts'
class TENAS:
    def __init__(self, dataset='imagenet-1k', data_path='/home/data/ImageNet_raw/', config_path=os.path.join(Settings.FOLDER,'datasets/configs/'), seed=0):
        if dataset == 'cifar10_valid_converged': 
            dataset = 'cifar10'
            self.layers = 20
            batch_size = 14
        else:
            self.layers = 14
            batch_size = 24
        print(dataset)
        self.dataset = dataset
        train_data, valid_data, xshape, self.class_num = get_datasets(dataset, data_path , -1)
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.search_space = get_search_spaces('cell', space)

        lrc_model = Linear_Region_Collector(input_size=(1000, 1, 3, 3), sample_batch=3, dataset=dataset, data_path=data_path, seed=seed)
        self.lrc_model = lrc_model

    def calculate_ntk_lr(self, genotype):
        Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
        genotype =  Genotype(normal=genotype[0], normal_concat=[2, 3, 4, 5], reduce=genotype[1], reduce_concat=[2, 3, 4, 5])
        model_config = dict({'name':'infer.nasnet-cifar',# 'DARTS-V1',
            'C': 1, 'N': 1, 'depth': 2, 'use_stem': True, 'stem_multiplier': 1,
            'num_classes': self.class_num,
            'space': self.search_space,
            'affine': True, 'track_running_stats': bool(1),
            'layers' : self.layers,
            'steps': 4,
            'multiplier': 4,
            'genotype':genotype
            })
        model_config_thin = dict({'name': 'infer.nasnet-cifar',# 'DARTS-V1',
            'C': 1, 'N': 1, 'depth': 2, 'use_stem': False, 'stem_multiplier': 1,
            'max_nodes': 4, 'num_classes': self.class_num,
            'space': self.search_space,
            'affine': True, 'track_running_stats': bool(1),
            'layers' : self.layers,
            'steps': 4,
            'multiplier': 4,
            'genotype': genotype                     
            })

        network = get_cell_based_tiny_net(model_config) # create the network from configurration
        network.drop_path_prob = 0
        network_thin = get_cell_based_tiny_net(model_config_thin) # create the network from configurration
        network_thin.drop_path_prob = 0
        init_model(network, 'kaiming_uniform'+"_fanout")  # for backward
        init_model(network_thin, 'kaiming_uniform'+"_fanin")  # for backward
        ntk = get_ntk_n(self.train_loader, [network], recalbn=0, train_mode=True, num_batch=1)
        self.lrc_model.reinit(models=[network_thin], seed=1)
        _lr = self.lrc_model.forward_batch_sample()
        torch.cuda.empty_cache()
        del network
        del network_thin
        return ntk, _lr


