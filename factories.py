from typing import Iterable
from typing import Tuple

import os

import torch
from torch import nn
import torchvision
from PIL import Image
import pandas as pd
import json


def read_circle(points):
    xmin = points[1]-points[3]
    ymin = points[2]-points[3]
    xmax = points[1]+points[3]
    ymax = points[2]+points[3]
    return [xmin, ymin, xmax, ymax]

def read_rect(points):
    xmin = points[1]
    ymin = points[2]
    xmax = points[1]+points[3]
    ymax = points[2]+points[4]
    return [xmin, ymin, xmax, ymax]

def read_ellipse(points):
    xmin = points[1]-points[3]
    ymin = points[2]-points[4]
    xmax = points[1]+points[3]
    ymax = points[2]+points[4]
    return [xmin, ymin, xmax, ymax]

def read_polygon(points):
    pts = points[1:]
    xs = pts[::2]
    ys = pts[1::2]
    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)
    return [xmin, ymin, xmax, ymax]

def raise_invalid_shape(points):
    shapes = ['None','dot','rectangle','circle','ellipse','line','lines','polygon','xrectangle','xcircle']
    raise ValueError('Inputshape "{}" ({}) is not rect/circle/ellipse/polygon and is therefore not handled'.format(shapes[points[0]],points[0]))
    
def read_shape(points):
    '''
    select shape according to xy-index in json and convert it to boundingbox
    '''
    shape_list = [raise_invalid_shape, # not specified
                  raise_invalid_shape, # dot: not suited for bbox
                  read_rect,
                  read_circle,
                  read_ellipse,
                  raise_invalid_shape, # line: not suited for bbox
                  raise_invalid_shape, # lines: not suited for bbox
                  read_polygon,
                  raise_invalid_shape, # xrectangular: not yet handled
                  raise_invalid_shape] # xcircle: not yet handled
    box = shape_list[points[0]](points)
    return box

class VIADataset(object):
    def __init__(self, root, transforms, json_path):
        self.root = root
        self.transforms = transforms
        self.classes = [{'Id':1, 'Name': 'Zelle'}]
        self.num_classes = 2 #for now, cell and background
        # load image dirs and labels from json
        #self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        with open(json_path) as json_file:
            self.imgs = json.load(json_file)

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx]['imgName'])
        mask_path = json.loads(self.imgs[idx]['viaProject'])['metadata']
        img = Image.open(img_path).convert("RGB")
        
        boxes = []
        for key, value in mask_path.items():
            # get boxes
            box = read_shape(value['xy'])
            boxes.append(box)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((len(mask_path),), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(mask_path),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class OptimizerFactory:
    def create_optimizer(self, name, parameters, args) -> torch.optim.Optimizer:
        if name == 'SGD':
            return torch.optim.SGD(parameters,
                                   lr=args.lr, momentum=args.momentum,
                                   weight_decay=args.weight_decay)

        elif name == 'RMSprop':
            return torch.optim.RMSprop(parameters, lr=args.lr,
                                       alpha=args.alpha, eps=args.eps,
                                       weight_decay=args.weight_decay, momentum = args.momentum)

        elif name == 'Adam':
            return torch.optim.Adam(parameters, lr=args.lr,
                                    eps=args.eps, weight_decay=args.weight_decay)

class DatasetFactory:
    def create_dataset(self, name, root, path_to_csv, transforms) -> Iterable[Tuple[Image.Image,int]]:
        if name == 'VIADataset':
            return VIADataset(root=root, json_path=path_to_csv, transforms=transforms)

        else:
            raise NotImplementedError(f'Dataset "{name}" not yet implemented')

class ModelFactory:
    def create_model(self, name, pretrained) -> nn.Module:
        if name == 'SmallCNN':
            raise NotImplementedError(f'model "{name}" not yet implemented')

        else:
            return torchvision.models.detection.__dict__[name](pretrained=pretrained)
        
class LossFactory:
    def create_loss(self, name) -> nn.Module:
        if name == 'MyLoss':
            raise NotImplementedError(f'loss "{name}" not yet implemented')

        else:
            return nn.modules.loss.__dict__[name]()
        
class LRSchedulerFactory:
    def create_scheduler(self, name, optimizer, args) -> torch.optim.lr_scheduler._LRScheduler:
        if name == 'MyScheduler':
            raise NotImplementedError(f'scheduler "{name}" not yet implemented')

        elif name == 'StepLR':
            return torch.optim.lr_scheduler.__dict__[name](optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        
        elif name == 'ExponentialLR':
            return torch.optim.lr_scheduler.__dict__[name](optimizer, gamma=args.lr_gamma)
        
        else:
            return torch.optim.lr_scheduler.__dict__[name]()
        
class TransformFactory:
    def create_transform(self, name, args) -> torch.optim.Optimizer:
        if name == 'Normalize':
            return torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
        
        elif name == 'RandomHorizontalFlip':
            return torchvision.transforms.RandomHorizontalFlip()

        elif name == 'RandomResizedCrop':
            return torchvision.transforms.RandomResizedCrop(tuple(args.input_size))
        
        elif name == 'ToTensor':
            return torchvision.transforms.ToTensor()
        
        elif name == 'Resize':
            return torchvision.transforms.Resize(tuple(args.input_size))
