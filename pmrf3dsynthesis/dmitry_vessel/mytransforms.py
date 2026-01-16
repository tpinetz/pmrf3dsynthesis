import typing

import numpy as np
import torch
import random
import math
import cv2

from scipy import ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from pmrf3dsynthesis.dmitry_vessel import loader_helper

class Transform:
    def __init__(self, transform_keys:list):
        self.transform_keys = transform_keys

    def __call__(self, data:dict):
        raise NotImplementedError()

class Compose:
    def __init__(self, transforms:list):
        self.transforms = transforms

    def __call__(self, data:dict):
        results = data
        for t in self.transforms:
            results = t(data)
        return results

#https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
def elastic_transform(image, alpha, sigma, order=3, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 3

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * (alpha/2.5)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')

    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))

    #print(dx[:100,0])

    return map_coordinates(image, indices, order=order, mode='reflect').reshape(shape)



def elastic_transform_ascan(image, alpha, sigma, order=3, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 3

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    shape2d = (shape[0],shape[2])
    displacement_map = random_state.rand(*shape2d) * 2 - 1
    displacement_map = gaussian_filter(displacement_map, sigma, mode="constant", cval=0) * alpha

    #dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    #dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    #dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * (alpha/2.5)

    dy = np.tile(displacement_map[:,None],(1,shape[1],1))

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')

    indices = np.reshape(x, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z, (-1, 1))

    #print(dx[:100,0])

    return map_coordinates(image, indices, order=order, mode='reflect').reshape(shape)

class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        results = args
        for t in self.transforms:
            results = t(*results)
        return results


class RandomCrop(Transform):
    def __init__(self, transform_keys:list, size):
        super(RandomCrop, self).__init__(transform_keys)
        self.size = np.array(size)


    def get_index(self,array,start_random):
        rel_size = np.minimum(self.size / np.array(array.shape), 1)
        start = (1. - rel_size) * start_random

        save_size_index = np.isnan(self.size)


        start = np.floor(start * array.shape)
        end = np.minimum(start + self.size, array.shape)

        start[save_size_index] = 0
        end[save_size_index] = np.array(array.shape)[save_size_index]

        index = tuple([slice(s, e) for s, e in zip(start.astype(np.int32), end.astype(np.int32))])

        return index

    def __call__(self, data:dict):

        if isinstance(data[self.transform_keys[0]], dict):
            shape = data[self.transform_keys[0]][0].shape
        else:
            shape = data[self.transform_keys[0]].shape

        #make cropping relative to the size

        start_random = np.random.random(size=(len(shape),))
        #.random.randint(0, np.maximum(max - crop, 1)) for max, crop in zip(shape,self.size)]


        for key in self.transform_keys:
            #assert len(data[key].shape) == len(shape)
            #assert np.all(data[key].shape == shape)

            if not key in data:
                continue

            if isinstance(data[key],dict):
                for subkey in data[key]:
                    index = self.get_index(data[key][subkey],start_random)
                    data[key][subkey] = data[key][subkey][index]
            else:
                index = self.get_index(data[key],start_random)
                data[key] = data[key][index].copy()

        return data

class BorderCrop(Transform):

    def __init__(self, transform_keys:list, size:typing.Tuple[int,...],mask_key:str):
        super(BorderCrop, self).__init__(transform_keys)
        self.mask_key = mask_key
        self.size = np.array(size)

    def get_index(self,array,center_random):
        rel_size = np.minimum(self.size / np.array(array.shape), 1)


        save_size_index = np.isnan(self.size)

        start = center_random - self.size//2
        start[start<0]=0
        end = start + self.size

        start[save_size_index] = 0
        end[save_size_index] = np.array(array.shape)[save_size_index]

        index = tuple([slice(s, e) for s, e in zip(start.astype(np.int32), end.astype(np.int32))])

        return index

    def __call__(self, data:typing.Dict[str,typing.Any]):
        if isinstance(data[self.transform_keys[0]], dict):
            shape = data[self.transform_keys[0]][0].shape
        else:
            shape = data[self.transform_keys[0]].shape

        mask_dict = data[self.mask_key]

        heatmap = np.stack(mask_dict.values(),axis=0).mean(axis=0)
        heatmap = ((heatmap > 0.05) == (heatmap < 0.95))

        if np.sum(heatmap) == 0:
            heatmap = np.zeros_like(heatmap)
            heatmap[:,self.size[1]:-self.size[1],:,self.size[3]:-self.size[3]] = 1

        density = heatmap/np.sum(heatmap)
        density1d = np.ravel(density)

        center = np.random.choice(a=len(density1d),size=1,replace=False,p=density1d)
        center = np.array(np.unravel_index(center,shape=density.shape)).flatten()

        for key in self.transform_keys + [self.mask_key]:
            #assert len(data[key].shape) == len(shape)
            #assert np.all(data[key].shape == shape)

            if not key in data:
                continue

            if isinstance(data[key],dict):
                for subkey in data[key]:
                    index = self.get_index(data[key][subkey],center)
                    data[key][subkey] = data[key][subkey][index]
            else:
                index = self.get_index(data[key],center)
                data[key] = data[key][index].copy()

        return data

import collections

class PadToShape(Transform):
    def __init__(self, transform_keys: list, shape, mode='minimum'):
        super(PadToShape, self).__init__(transform_keys)
        self.shape = shape
        self.mode = mode

    def __call__(self, data: dict):

        for key in self.transform_keys:
            #assert len(data[key].shape) == len(data_shape)
            #assert np.all(data[key].shape == data_shape)


            if isinstance(data[key],(dict,collections.OrderedDict)):
                if data[key]:
                    data_shape = data[key][list(data[key].keys())[0]].shape
                    for subkey in data[key]:
                        padded, (pad_left, pad_right) = loader_helper.pad_to_shape(data[key][subkey], self.shape, mode=self.mode)
                        data['padding' + key] = (pad_left, pad_right)
                        data[key][subkey] = padded
                    data['shape_old' + key] = data_shape
                    data['shape_new' + key] = self.shape
            else:

                data_shape = data[key].shape
                padded,(pad_left, pad_right) = loader_helper.pad_to_shape(data[key],self.shape, mode=self.mode)

                data['shape_old'+key] = data_shape
                data['shape_new'+key] = self.shape
                data['padding'+key] = (pad_left, pad_right)
                data[key] = padded

        return data

class PadSidesByValues(Transform):
    def __init__(self, transform_keys: list, shape, mode='minimum'):
        super(PadSidesByValues, self).__init__(transform_keys)
        self.shape = shape
        self.mode = mode

    def __call__(self, data: dict):

        for key in self.transform_keys:
            if isinstance(data[key],(dict,collections.OrderedDict)):
                if data[key]:
                    data_shape = data[key][list(data[key].keys())[0]].shape
                    for subkey in data[key]:
                        padded = np.pad(data[key][subkey],tuple(zip(self.shape, self.shape)),mode=self.mode)
                        data['padding' + key] = (self.shape, self.shape)
                        data[key][subkey] = padded
                        data['shape_old' + key] = data_shape
                        data['shape_new' + key] = padded.shape
            else:

                data_shape = data[key].shape
                padded = np.pad(data[key],tuple(zip(self.shape, self.shape)),mode=self.mode)

                data['shape_old'+key] = data_shape
                data['shape_new'+key] = padded.shape
                data['padding'+key] = (self.shape, self.shape)
                data[key] = padded

        return data

class DivByArea(Transform):
    def __init__(self, transform_keys: list):
        super(DivByArea, self).__init__(transform_keys)

    def __call__(self, data: dict):


        pixel_area = data['affine'][0, 2] * data['affine'][2, 0]

        for key in self.transform_keys:

            if isinstance(data[key],(dict,collections.OrderedDict)):
                if data[key]:
                    for subkey in data[key]:
                        data[key][subkey] = data[key][subkey] / pixel_area
            else:
                data[key] = data[key] / pixel_area

        return data

class ComputeArea(Transform):
    def __init__(self, transform_keys: list):
        super(ComputeArea, self).__init__(transform_keys)

    def __call__(self, data: dict):


        pixel_area = data['pixel_area']

        data['areas'] = {k: v.sum()* pixel_area if v is not None else None for k, v in data['masks'].items()}

        return data

class DropMasks(Transform):
    def __init__(self, transform_keys:list=None):
        super(DropMasks, self).__init__(transform_keys)

    def __call__(self, data: dict):

        real_masks = [k for k,v in data['masks'].items() if v is not None]

        if len(real_masks) > 2:
            for i in real_masks[1:-1]:
                if i != 360:
                    data['masks'][i] = None

        return data


class DivByArea2D(Transform):
    def __init__(self, transform_keys: list):
        super(DivByArea2D, self).__init__(transform_keys)

    def __call__(self, data: dict):


        pixel_area = data['affine'][0, 1] * data['affine'][1, 0]

        for key in self.transform_keys:

            if isinstance(data[key],(dict,collections.OrderedDict)):
                if data[key]:
                    for subkey in data[key]:
                        data[key][subkey] = data[key][subkey] / pixel_area
            else:
                data[key] = data[key] / pixel_area

        return data


class AreaDiff(Transform):
    def __init__(self,):
        super(AreaDiff, self).__init__([])

    def __call__(self, data: dict):

        diff = data['areas'][12.] - data['areas'][0.]
        diff = diff if diff > 0 else np.array([0])
        diff = diff.reshape((1,)).astype(np.float32)

        data['diff'] = diff
        #self.denominator_key = 'image'
        #data['diff'] = diff*data[self.denominator_key].shape[1] * data[self.denominator_key].shape[3]
        return data

class DivBySize(Transform):
    def __init__(self, transform_keys: list, denominator_key):
        super(DivBySize, self).__init__(transform_keys)
        self.denominator_key = denominator_key

    def __call__(self, data: dict):

        total_pixels = data[self.denominator_key].shape[1] * data[self.denominator_key].shape[3]

        for key in self.transform_keys:

            if isinstance(data[key],(dict,collections.OrderedDict)):
                if data[key]:
                    for subkey in data[key]:
                        data[key][subkey] = data[key][subkey] / total_pixels
            else:
                data[key] = data[key] / total_pixels

        return data

class DivBySize2D(Transform):
    def __init__(self, transform_keys: list, denominator_key):
        super(DivBySize2D, self).__init__(transform_keys)
        self.denominator_key = denominator_key

    def __call__(self, data: dict):

        total_pixels = data[self.denominator_key].shape[1] * data[self.denominator_key].shape[2]

        for key in self.transform_keys:

            if isinstance(data[key],(dict,collections.OrderedDict)):
                if data[key]:
                    for subkey in data[key]:
                        data[key][subkey] = data[key][subkey] / total_pixels
            else:
                data[key] = data[key] / total_pixels

        return data

class Mean(Transform):
    def __init__(self, transform_keys: list):
        super(Mean, self).__init__(transform_keys)

    def __call__(self, data: dict):

        for key in self.transform_keys:

            C = data[key].shape[0]

            data[key+'_mean'] = np.mean(data[key].reshape((C,-1)),axis=1,dtype=np.float32)

        return data

class PadToRate(Transform):
    def __init__(self, transform_keys: list, rate:list, dim:list=None, mode='minimum'):
        super(PadToRate, self).__init__(transform_keys)
        self.rate = np.array(rate)
        self.dim = dim
        self.mode = mode

    def pad_to_rate(self,array,key):
        data_shape = array.shape
        if self.dim is None:
            self.dim = list(range(len(data_shape)))

        new_shape = [loader_helper.closest_to_k(i, r) if idx in self.dim else i for idx, (i, r) in
                     enumerate(zip(data_shape, self.rate))]
        new_shape = tuple(new_shape)

        padded, (pad_left, pad_right) = loader_helper.pad_to_shape(array, new_shape,mode = self.mode)


        desc = {}
        desc['shape_old' + key] = data_shape
        desc['shape_new' + key] = new_shape
        desc['padding' + key] = (pad_left, pad_right)
        return padded, desc

    def __call__(self, data: dict):
        #data_shape = data[self.transform_keys[0]].shape

        for key in self.transform_keys:
            #assert len(data[key].shape) == len(data_shape)
            #assert np.all(data[key].shape == data_shape)

            if not key in data:
                continue
            if isinstance(data[key],dict):
                for subkey in data[key]:
                    array, desc = self.pad_to_rate(data[key][subkey], key)
                    data[key][subkey] = array
                    data.update(desc)
            else:
                array, desc = self.pad_to_rate(data[key],key)
                data[key] = array
                data.update(desc)

        return data

class RandomMirror(Transform):
    def __init__(self, transform_keys:list, dimensions:list):
        super(RandomMirror, self).__init__(transform_keys)
        self.dimensions = dimensions

    def flip(self, image, p):

        index = [slice(0, size) for size in image.shape]

        for i in self.dimensions:
            if p[i] < 0.5:
                index[i] = slice(-1, -image.shape[i] - 1, -1)

        index = tuple(index)

        return image[index].copy()

    def __call__(self, data:dict):

        #print('TEST,crop',shape,self.size,[(max - crop or 1) for max, crop in zip(shape,self.size)])

        if isinstance(data[self.transform_keys[0]], dict):
            dim = len(data[self.transform_keys[0]][0].shape)
        else:
            dim = len(data[self.transform_keys[0]].shape)

        p = np.random.random(dim)

        for key in self.transform_keys:

            if not key in data:
                continue
            if isinstance(data[key],dict):
                for subkey in data[key]:
                    if data[key][subkey] is not None:
                        data[key][subkey] = self.flip(data[key][subkey],p)

            else:
                data[key] = self.flip(data[key],p)

        return data

class ScaleAffine(Transform):
    def __init__(self, factor:list):
        super(ScaleAffine, self).__init__([])
        self.factor = factor

    def __call__(self, data:dict):

        #print('TEST,crop',shape,self.size,[(max - crop or 1) for max, crop in zip(shape,self.size)])

        data['affine'] = np.dot(data['affine'],np.diag(self.factor))
        if 'pixel_area' in data:
            data['pixel_area'] = data['pixel_area']*np.prod(self.factor)
        if 'pixels' in data:
            data['pixels'] = data['pixels'] / np.prod(self.factor)

        return data

class Scale(Transform):
    def __init__(self, transform_keys:list, factor:list, order=1):
        super(Scale, self).__init__(transform_keys)
        self.factor = factor
        self.order=order

    def __call__(self, data:dict):

        #print('TEST,crop',shape,self.size,[(max - crop or 1) for max, crop in zip(shape,self.size)])

        for key in self.transform_keys:
            if not key in data:
                continue

            if isinstance(data[key],dict):
                for subkey in data[key]:
                    data[key][subkey] = ndimage.zoom(data[key][subkey],zoom=self.factor,order=self.order)

            else:
                data[key] = ndimage.zoom(data[key],zoom=self.factor,order=self.order)

        return data

class ZScoreNormalization(Transform):
    def __init__(self, transform_keys:list, axis):
        super(ZScoreNormalization, self).__init__(transform_keys)
        self.axis = axis

    def __call__(self, data:dict):
        #mean = 9310.
        #std = 8759
        for key in self.transform_keys:

            if isinstance(data[key],dict):
                for subkey in data[key]:
                    mean = data[key][subkey].mean(axis=self.axis, keepdims=True)
                    std = data[key][subkey].std(axis=self.axis, keepdims=True)

                    data[key][subkey] = (data[key][subkey] - mean) / (std + 1e-5)
            else:
                mean = data[key].mean(axis=self.axis, keepdims=True)
                std = data[key].std(axis=self.axis, keepdims=True)

                data[key] = (data[key] - mean) / (std + 1e-5)

        return data

class NormalizationByValue(Transform):
    def __init__(self, transform_keys:list, bias, scale):
        super(NormalizationByValue, self).__init__(transform_keys)
        self.bias = bias
        self.scale = scale

    def __call__(self, data:dict):
        #mean = 9310.
        #std = 8759
        for key in self.transform_keys:

            if isinstance(data[key],dict):
                for subkey in data[key]:
                    data[key][subkey] = (data[key][subkey] - self.bias) / self.scale
            else:
                data[key] = (data[key] - self.bias) / (self.scale + 1e-5)

        return data

class IntensityShift(Transform):
    def __init__(self, transform_keys:list, min:float = -0.6, max:float = 0.6):
        super(IntensityShift, self).__init__(transform_keys)
        self.min = min
        self.max = max

    def __call__(self, data:dict):

        for key in self.transform_keys:

            if isinstance(data[key], dict):
                for subkey in data[key]:
                    data[key][subkey] = data[key][subkey] + random.uniform(self.min, self.max)

            else:
                data[key] = data[key] + random.uniform(self.min, self.max)

        return data

class ContrastAugmentation(Transform):
    def __init__(self, transform_keys: list, min: float = 0.6, max: float = 1.4):
        super(ContrastAugmentation, self).__init__(transform_keys)
        self.min = min
        self.max = max

    def __call__(self, data: dict):
        for key in self.transform_keys:
            if isinstance(data[key], dict):
                for subkey in data[key]:
                    data[key][subkey] = data[key][subkey] * random.uniform(self.min, self.max)

            else:
                data[key] = data[key] * random.uniform(self.min, self.max)

        return data

class AddNoiseAugmentation(Transform):
    def __init__(self, transform_keys: list, dim, mu: float = 0.0, sigma: float = 1.0):
        super(AddNoiseAugmentation, self).__init__(transform_keys)
        self.mu = mu
        self.sigma = sigma
        self.dim = dim

    def __call__(self, data: dict):
        for key in self.transform_keys:

            if isinstance(data[key],dict):
                for subkey in data[key]:
                    shape = [i if idx in self.dim else 1 for idx, i in enumerate(data[key][subkey].shape)]
                    noise = np.random.normal(self.mu, self.sigma, size=shape)

                    data[key][subkey] = data[key][subkey] + noise  # random.uniform(self.min, self.max)

            else:
                shape = [i if idx in self.dim else 1 for idx, i in enumerate(data[key].shape)]
                noise = np.random.normal(self.mu, self.sigma, size=shape)

                data[key] = data[key] + noise  # random.uniform(self.min, self.max)


class MulNoiseAugmentation(Transform):
    def __init__(self, transform_keys: list, dim, mu: float = 1.0, sigma: float = 1.0):
        super(MulNoiseAugmentation, self).__init__(transform_keys)
        self.mu = mu
        self.sigma = sigma
        self.dim = dim

    def __call__(self, data: dict):
        for key in self.transform_keys:


            if isinstance(data[key],dict):
                for subkey in data[key]:
                    shape = [i if idx in self.dim else 1 for idx, i in enumerate(data[key][subkey].shape)]
                    noise = np.random.normal(self.mu, self.sigma, size=shape)

                    data[key][subkey] = data[key][subkey] * noise  # random.uniform(self.min, self.max)

            else:
                shape = [i if idx in self.dim else 1 for idx, i in enumerate(data[key].shape)]
                noise = np.random.normal(self.mu, self.sigma, size=shape)

                data[key] = data[key] * noise  # random.uniform(self.min, self.max)




class InPaintEnface(Transform):
    def __init__(self, transform_keys:list,nframes:int=4,box_shape:list=[4,4],min_box_shape:list=[2,2]):
        super(InPaintEnface, self).__init__(transform_keys)
        self.nframes = nframes
        self.box_shape = box_shape
        self.min_box_shape = min_box_shape

    def __call__(self, data:dict):

        for key in self.transform_keys:

            for i in range(self.nframes):
                shape = data[key].shape
                dx = random.randrange(self.min_box_shape[0],self.box_shape[0])
                dy = random.randrange(self.min_box_shape[1],self.box_shape[1])

                xbox = random.randrange(0, shape[1] - dx)
                ybox = random.randrange(0, shape[3] - dy)

                data[key][:,xbox:xbox+dx,:,
                          ybox:ybox+dy,] = np.random.normal(0,1,size=(shape[0],dx,shape[2],dy))

class ToTensorDict(Transform):
    def __init__(self, transform_keys: list):
        super(ToTensorDict, self).__init__(transform_keys)

    def __call__(self, data: dict):
        for key in self.transform_keys:

            if not key in data:
                continue

            if isinstance(data[key],dict):
                for subkey in data[key]:
                    if data[key][subkey] is not None:
                        data[key][subkey] = torch.from_numpy(data[key][subkey]).float()

            else:
                data[key] = torch.from_numpy(data[key]).float()

        return data


class NoiseAugmentation(Transform):
    def __init__(self, transform_keys, sigma = 2.0, prob = 0.2):
        super(NoiseAugmentation, self).__init__(transform_keys)
        self.sigma = sigma
        self.prob = prob

    def __call__(self, data: dict):

        for key in self.transform_keys:
            if random.random() < self.prob:
                data[key] = data[key] + np.random.normal(0, self.sigma, size=data[key].shape)

        return data

class InPaint(Transform):
    def __init__(self, transform_keys:list,nframes:int=4,box_shape_min:list=[1,4,4,4],box_shape_max:list=[1,2,2,2]):
        super(InPaint, self).__init__(transform_keys)
        self.nframes = nframes
        self.box_shape_min = box_shape_min
        self.box_shape_max = box_shape_max

    def __call__(self, data:dict):

        for key in self.transform_keys:

            for i in range(self.nframes):

                box_shape = np.array(self.box_shape_min) + np.random.random(size=(len(self.box_shape_max))) * (np.array(self.box_shape_max) - np.array(self.box_shape_min))
                box_shape = np.floor(box_shape).astype(np.int32)

                start = np.floor(np.random.random(size=len(data[key].shape))* (np.array(data[key].shape) - box_shape)).astype(np.int32)

                index = [slice(s, s+b, 1) for s,b in zip(start, box_shape)]

                data[key][tuple(index)] = random.uniform(-1,1)

        return data


class ComputeAreas(Transform):
    def __init__(self, transform_keys):
        super(ComputeAreas, self).__init__(transform_keys)
        assert(len(self.transform_keys)==1)

    def __call__(self, data: dict):

        areas = [m.reshape(m.shape[0],-1).sum(axis=-1) * data['pixel_area'] if m is not None else None for m in data[self.transform_keys[0]].values()]
        data['areas'] = {k:a for k,a in zip(data['visit_since_baseline'], areas) if a is not None }#areas

        return data


class AScanShift:
    def __init__(self, sigma = 5, alpha = 2):
        self.sigma = sigma
        self.alpha = alpha

    def __call__(self, *args):

        results = []

        for a in args:
            assert len(a.shape) == 3 or len(a.shape) == 4

            if a.dtype == np.float32:
                #mask = np.random.uniform(0, 1, size=a.shape) < self.prob
                for i in range(a.shape[0]):
                    a[i] = elastic_transform_ascan(a[i], self.alpha, self.sigma)

            results.append(a)

        return tuple(results)


class RandomAffineTransform:
    def __init__(self, theta = (-15, 15), x_scale = (0.8,1.2), y_scale = (0.8,1.2)):
        self.theta = theta
        self.x_scale = x_scale
        self.y_scale = y_scale

    def __call__(self, *args):

        results = []

        theta = np.random.uniform(self.theta[0],self.theta[1])
        cos_t = math.cos(theta * math.pi / 180)
        x_scale = np.random.uniform(self.x_scale[0],self.x_scale[1])
        y_scale = np.random.uniform(self.y_scale[0],self.y_scale[1])

        rotation_matrix = np.array([[math.cos(theta), - math.sin(theta)],[math.sin(theta), math.cos(theta)]])
        scale_matrix = np.array([[x_scale,0],[0,y_scale]])

        transform_matrix = np.matmul(rotation_matrix,scale_matrix)

        for a in args:
            assert len(a.shape) == 3

            if a.shape[1] == 1:
                mat = (1, 1 , np.abs(cos_t))
                a = ndimage.affine_transform(a, matrix=mat, offset=(0, 0, a.shape[2] * (1 - np.abs(cos_t)) / 2),
                                             mode='constant', cval=0)
            else:
                a = ndimage.rotate(a, theta, axes=(1, 2), reshape=False)

            results.append(a)

        return tuple(results)


class LineAug2D:
    def __init__(self, flag):
        self.flag = flag

    def __call__(self, *args):

        assert(len(self.flag) == len(args))

        results = []

        for a, f in zip(args,self.flag):
            assert len(a.shape) == 3
            if not f:
                results.append(a)
                continue

            assert(a.shape[1]==1)

            img = a[:,0]

            mask = np.zeros_like(img)

            start = np.random.uniform(0, mask.shape)
            end = np.random.uniform(0, mask.shape)

            start[0] = 0
            if np.random.random() > 0.5:
                end[1] = mask.shape[1]
            else:
                end[0] = mask.shape[0]

            mask = cv2.line(mask, tuple(start[::-1].astype(int)), tuple(end[::-1].astype(int)), 255, 3,
                            cv2.LINE_AA)

            img[mask>0] = img.min()

            results.append(img[:,None])

        return tuple(results)


class ToTensor:
    def __call__(self, *args):

        results = []

        for a in args:
            img = torch.from_numpy(a).float()
            results.append(img)

        return tuple(results)