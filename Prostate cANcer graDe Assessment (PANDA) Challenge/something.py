#=================================== To split an images to list of small-partitions ====================
import os
import cv2
import numpy as np
import skimage
from skimage.transform import rescale, resize

def get_tiles_with_id(data_dir, mask_dir, img_id, level, mode=0, n_tiles = 81, ops = 256):
        """
            Input: 
                    - img_id (str): image_id from the train dataset
                    - level (int): an integer in {0, 1, 2} corresponding to the level_downsamples {1, 4, 16}
                    - mode (int) : define the quantities of pad_height & pad_width
                    - n_tiles (int): number of tiles (must be a squared_number)
                    - ops (int) : output_size of each image
            return: 
                    - list of img_data_tiles
                    - img_mask
                    - bool
        """
        tile_size = int(256 / 2**(2*level))
        data_img = skimage.io.MultiImage(os.path.join(data_dir, f'{img_id}.tiff'))[level]
        mask_img = skimage.io.MultiImage(os.path.join(mask_dir, f'{img_id}_mask.tiff'))[level]
        
        image_data_ls = []; image_mask_ls = []
        
        h, w = data_img.shape[:2]
        pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
        pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)
        
        img2_dt_ = np.pad(data_img,[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2,pad_w - pad_w//2], [0,0]], constant_values = 255)
        img2_ms_ = np.pad(mask_img,[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2,pad_w - pad_w//2], [0,0]], constant_values = mask_img.max())
        
        img3_dt_ = img2_dt_.reshape(img2_dt_.shape[0] // tile_size, tile_size,
                                    img2_dt_.shape[1] // tile_size, tile_size,
                                    3 )
        img3_ms_ = img2_ms_.reshape(img2_ms_.shape[0] // tile_size, tile_size,
                                    img2_ms_.shape[1] // tile_size, tile_size,
                                    3 )
        
        img3_dt_ = img3_dt_.transpose(0,2,1,3,4).reshape(-1, tile_size, tile_size,3)
        img3_ms_ = img3_ms_.transpose(0,2,1,3,4).reshape(-1, tile_size, tile_size,3)
        
        n_tiles_with_info = (img3_dt_.reshape(img3_dt_.shape[0],-1).sum(1) < tile_size ** 2 * 3 * 255).sum()
        
        if len(data_img) < n_tiles:
            img3_dt_ = np.pad(img3_dt_,[[0,N - len(img3_dt_)],[0,0],[0,0],[0,0]], constant_values=255)
            img3_ms_ = np.pad(img3_ms_,[[0,N - len(img3_ms_)],[0,0],[0,0],[0,0]], constant_values = mask_img.max())
            
        idxs_dt_ = np.argsort(img3_dt_.reshape(img3_dt_.shape[0],-1).sum(-1))[:n_tiles]    
        
        img3_dt_ = img3_dt_[idxs_dt_]
        img3_ms_ = img3_ms_[idxs_dt_]
        
        
        for i in range(len(img3_dt_)):
            img4_dt_ = cv2.resize(img3_dt_[i], (ops, ops))
            image_data_ls.append({'img':img4_dt_, 'idx':i})
            img4_ms_ = cv2.resize(img3_ms_[i], (ops, ops))
            image_mask_ls.append({'img':img4_ms_, 'idx':i})
            del img4_dt_, img4_ms_
        return image_data_ls, image_mask_ls, n_tiles_with_info >= n_tiles

def get_image(img_id_ls, level):
        """
            Input: 
                    - img_id (str): image_id from the train dataset
                    - level (int): an integer in {0, 1, 2} corresponding to the level_downsamples {1, 4, 16}
            return: 
                    - list of img_data & img_mask
        """
        
        img_data_list, img_mask_list = [], []
        for img_id in img_id_ls:
            data_img = skimage.io.MultiImage(os.path.join(data_dir, f'{img_id}.tiff'))[level]
            mask_img = skimage.io.MultiImage(os.path.join(mask_dir, f'{img_id}_mask.tiff'))[level]
            
            img_data_list.append(data_img) 
            img_mask_list.append(mask_img)
            del data_img, mask_img
        return img_data_list, img_mask_list, img_id_ls

def get_tiles_image(data_img, mask_img, level=2, mode=0, n_tiles = 36, ops = 256):
        """
            Input: 
                    - img_id (str): image_id from the train dataset
                    - level (int): an integer in {0, 1, 2} corresponding to the level_downsamples {1, 4, 16}
                    - mode (int) : define the quantities of pad_height & pad_width
                    - n_tiles (int): number of tiles (must be a squared_number)
                    - ops (int) : output_size of each image
            return: 
                    - list of img_data_tiles
                    - img_mask
                    - bool
        """
        tile_size = 256
        h, w = data_img.shape[: 2]
        data_img = cv2.resize(data_img, (16*h, 16*w))
        mask_img = cv2.resize(mask_img, (16*h, 16*w))
        
        image_data_ls = []; image_mask_ls = []
        
        h, w = data_img.shape[:2]
        pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
        pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)
        
        img2_dt_ = np.pad(data_img,[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2,pad_w - pad_w//2], [0,0]], constant_values = 255)
        img2_ms_ = np.pad(mask_img,[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2,pad_w - pad_w//2], [0,0]], constant_values = mask_img.max())
        
        img3_dt_ = img2_dt_.reshape(img2_dt_.shape[0] // tile_size, tile_size,
                                    img2_dt_.shape[1] // tile_size, tile_size,
                                    3 )
        img3_ms_ = img2_ms_.reshape(img2_ms_.shape[0] // tile_size, tile_size,
                                    img2_ms_.shape[1] // tile_size, tile_size,
                                    3 )
        
        img3_dt_ = img3_dt_.transpose(0,2,1,3,4).reshape(-1, tile_size, tile_size,3)
        img3_ms_ = img3_ms_.transpose(0,2,1,3,4).reshape(-1, tile_size, tile_size,3)
        
        n_tiles_with_info = (img3_dt_.reshape(img3_dt_.shape[0],-1).sum(1) < tile_size ** 2 * 3 * 255).sum()
        
        if len(data_img) < n_tiles:
            img3_dt_ = np.pad(img3_dt_,[[0,N - len(img3_dt_)],[0,0],[0,0],[0,0]], constant_values=255)
            img3_ms_ = np.pad(img3_ms_,[[0,N - len(img3_ms_)],[0,0],[0,0],[0,0]], constant_values = mask_img.max())
            
        idxs_dt_ = np.argsort(img3_dt_.reshape(img3_dt_.shape[0],-1).sum(-1))[:n_tiles]    
        
        img3_dt_ = img3_dt_[idxs_dt_]
        img3_ms_ = img3_ms_[idxs_dt_]
        
        
        for i in range(len(img3_dt_)):
            img4_dt_ = cv2.resize(img3_dt_[i], (ops, ops))
            image_data_ls.append({'img':img4_dt_, 'idx':i})
            img4_ms_ = cv2.resize(img3_ms_[i], (ops, ops))
            image_mask_ls.append({'img':img4_ms_, 'idx':i})
            del img4_dt_, img4_ms_
        return image_data_ls, image_mask_ls, n_tiles_with_info >= n_tiles
  
#=================================== To build U-Net neural network ====================

import torch
from torch import nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv'):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out

#=============================================
from collections import OrderedDict
import math

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class Bottleneck(nn.Module):
  """
  Base class for bottlenecks that implements `forward()` method.
  """
  def forward(self, x):
      residual = x

      out = self.conv1(x)     ## conv_olution 
      out = self.bn1(out)     ## bottle_neck
      out = self.relu(out)

      out = self.conv2(out)
      out = self.bn2(out)
      out = self.relu(out)

      out = self.conv3(out)
      out = self.bn3(out)

      if self.downsample is not None:
          residual = self.downsample(x)

      out = self.se_module(out) + residual
      out = self.relu(out)

      return out

class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4
    def __init__(self, inplanes, planes, groups, reduction, stride = 1, downsample = None):

        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size = 3,
                               stride=stride, padding = 1, groups = groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

class SEResNetBottleneck(Bottleneck):
  """
  ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
  implementation and uses `stride=stride` in `conv1` and not in `conv2`
  (the latter is used in the torchvision implementation of ResNet).
  """
  expansion = 4

  def __init__(self, inplanes, planes, groups, reduction, stride = 1, downsample = None):
      super(SEResNetBottleneck, self).__init__()
      self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 1, bias = False, stride = stride)
      self.bn1 = nn.BatchNorm2d(planes)
      self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, padding = 1,
                             groups = groups, bias = False)
      self.bn2 = nn.BatchNorm2d(planes)
      self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size = 1, bias = False)
      self.bn3 = nn.BatchNorm2d(planes * 4)
      self.relu = nn.ReLU(inplace = True)
      self.se_module = SEModule(planes * 4, reduction=reduction)
      self.downsample = downsample
      self.stride = stride

class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride = 1,
                 downsample = None, base_width = 4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size = 3, stride = stride,
                               padding = 1, groups = groups, bias = False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace = True)
        self.se_module = SEModule(planes * 4, reduction = reduction)
        self.downsample = downsample
        self.stride = stride

class SENet(nn.Module):
    def __init__(self, block, layers, groups, reduction, dropout_p = 0.2,
                 inplanes = 128, input_3x3 = True, downsample_kernel_size = 3,
                 downsample_padding = 1, num_classes = 1000):
        
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [ ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)),
                                ('bn1', nn.BatchNorm2d(64)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,bias=False)),
                                ('bn2', nn.BatchNorm2d(64)),
                                ('relu2', nn.ReLU(inplace=True)),
                                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)),
                                ('bn3', nn.BatchNorm2d(inplanes)),
                                ('relu3', nn.ReLU(inplace=True)) ]
        else:
            layer0_modules = [ ('conv1', nn.Conv2d(3, inplanes, kernel_size = 7, stride = 2, 
                                                   padding = 3, bias = False)),
                              ('bn1', nn.BatchNorm2d(inplanes)), 
                              ('relu1', nn.ReLU(inplace = True)) ]
        # create the layers
        # To preserve compatibility with Caffe weights `ceil_mode=True` is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride = 2, ceil_mode = True)))
        
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        
        self.layer1 = self._make_layer( block, planes = 64, blocks = layers[0],
                                        groups = groups, reduction = reduction,
                                       downsample_kernel_size = 1, 
                                       downsample_padding = 0 )
        
        self.layer2 = self._make_layer( block, planes = 128, blocks=layers[1], stride=2,
                                        groups = groups, reduction=reduction,
                                        downsample_kernel_size = downsample_kernel_size,
                                        downsample_padding = downsample_padding )

        self.layer3 = self._make_layer( block, planes = 256, blocks = layers[2], stride=2,
                                        groups = groups, reduction = reduction,
                                        downsample_kernel_size = downsample_kernel_size,
                                        downsample_padding = downsample_padding )
        
        self.layer4 = self._make_layer( block, planes = 512, blocks=layers[3], stride = 2,
                                        groups = groups, reduction=reduction,
                                        downsample_kernel_size = downsample_kernel_size,
                                        downsample_padding = downsample_padding )
        
        self.avg_pool = nn.AvgPool2d(7, stride = 1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential( nn.Conv2d( self.inplanes, planes * block.expansion,
                                                   kernel_size = downsample_kernel_size, stride = stride,
                                                   padding = downsample_padding, bias = False),
                                        nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x
def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = config.pretrained_settings['se_resnext50_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = config.pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model

class CustomSEResNeXt(nn.Module):

    def __init__(self, model_name='se_resnext50_32x4d'):
        assert model_name in ('se_resnext50_32x4d')
        super().__init__()
        
        self.model = se_resnext50_32x4d(pretrained=None)
        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, config.CLASSES)
        
    def forward(self, x):
        x = self.model(x)
        return x

class PandaDataset(Dataset):
    def __init__(self, images, img_height, img_width):
        self.images = images
        self.img_height = img_height
        self.img_width = img_width
        
        # we are in validation part
        self.aug = albumentations.Compose([
            albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], always_apply=True)
        ])

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):

        img_name = self.images[idx]
        img_path = os.path.join(data_dir, f'{img_name}.tiff')

        img = skimage.io.MultiImage(img_path)
        img = cv2.resize(img[-1], (512, 512))
        save_path =  f'{img_name}.png'
        cv2.imwrite(save_path, img)
        img = skimage.io.MultiImage(save_path)
            
        img = cv2.resize(img[-1], (self.img_height, self.img_width))

        img = Image.fromarray(img).convert("RGB")
        img = self.aug(image=np.array(img))["image"]
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)

        return { 'image': torch.tensor(img, dtype=torch.float) }
