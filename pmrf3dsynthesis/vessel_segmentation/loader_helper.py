import nibabel as nib
import numpy as np
import os
import torch
import pydicom
import re
import pandas as pd

import lxml.etree

from scipy import ndimage

class DuplicateError(Exception):
    def __init__(self, key, message):
        self.key = key
        self.message = message

def get_key(parent_key, key):
    if parent_key:
        key = parent_key + '.' + key
    return key

# lookup table for naming rules
naming_rule = {
    'Visit.Scan.ScanProperties.ScanProperty': 'name',
    'PatientProperties.PatientProperty': 'name',
    'Visit.VisitProperties.VisitProperty': 'name'
}




def average_outputs(outputs, dtype):
    if isinstance(outputs,list) and dtype==dict:
        keys = outputs[0].keys()
        return {key: average_outputs([d[key] for d in outputs],dtype=type(outputs[0][key])) for key in keys}

    elif isinstance(outputs,list) and dtype==str:
        return outputs[0]
    elif isinstance(outputs, list) and dtype == torch.Tensor:
        return torch.mean(torch.stack(outputs,dim=0),dim=0)
    else:
        assert()


def bootstrap_ci(df, variable, repetitions=1000, alpha=0.05, random_state=None):
    df = df[variable]
    bootstrap_sample_size = len(df)

    means = []
    for i in range(repetitions):
        bootstrap_sample = df.sample(n=bootstrap_sample_size, replace=True, random_state=random_state)
        mean = bootstrap_sample.mean()
        means.append(mean)  # confidence interval
    left = np.percentile(means, alpha / 2 * 100)
    right = np.percentile(means, 100 - alpha / 2 * 100)
    return left, right

def get_dict_from_root(root, parent_key, mydict):
    for tree in root:
        key = tree.tag
        key = get_key(parent_key, key)
        if tree.getchildren():
            mydict = get_dict_from_root(tree, key, mydict)
        else:
            if key in naming_rule:
                key = key + '.' + str(tree.attrib['name'])
            if key in mydict: #key in mydict
                raise DuplicateError(key=key, message='Duplicate key has been found')

            mydict[key] = str(tree.text)


    return mydict

def read_slo(slo_header, oct_header):
        oct_spacing = np.array(list(map(float, [oct_header.PixelSpacing[0],
                                                oct_header.SpacingBetweenSlices])))
        oct_size = np.array([oct_header.Columns,
                             oct_header.NumberOfFrames])

        frameLocations = oct_header.OphthalmicFrameLocationSequence
        slo_spacing = np.array(list(map(float, slo_header.PixelSpacing)))

        bscanCoordinates = []
        for bscan in frameLocations:
            line = str(bscan).split('FL: ')[1]
            line = list(map(float, line.strip("[]").split(',')))
            bscanCoordinates.append(line)
        bscanCoordinates = np.array(bscanCoordinates)

        # first horizontal position of every bscan [mm]
        x0, _, _, _ = bscanCoordinates[0]
        horizontal_positions = [int((x0 + i * oct_spacing[0]) / slo_spacing[0]) \
                                for i in range(oct_size[0])]
        vertical_positions = list(map(int, bscanCoordinates[:, 2] / slo_spacing[1]))

        x_min, x_max = np.min(horizontal_positions), np.max(horizontal_positions)  # ascans
        y_min, y_max = np.min(vertical_positions), np.max(vertical_positions)  # bscans

        slo_header.SamplesPerPixel = 1
        slo_image = slo_header.pixel_array
        slo_image_min = slo_image.min()
        slo_image_max = slo_image.max()
        slo_image = (slo_image - slo_image_min) / (slo_image_max - slo_image_min)
        slo_image = (slo_image * 255).astype(np.uint8)[:, :, None]
        slo_image = np.tile(slo_image, (1, 1, 3))

        return slo_image, x_min, x_max, y_min, y_max

def read_oct(oct_header):
    oct_header.SamplesPerPixel = 1

    array = oct_header.pixel_array.astype(np.float32)
    spacing = oct_header.PixelSpacing
    slice_spacing = oct_header.SpacingBetweenSlices
    spacing = np.array([slice_spacing, spacing[1], spacing[0]])

    return  array, spacing

def read_metadata(metadata_path):
    tree = lxml.etree.parse(metadata_path)
    root = tree.getroot()

    return get_dict_from_root(root, None, {})

def read_dicom(oct_path):
    dataset = pydicom.read_file(oct_path)
    dataset.SamplesPerPixel = 1

    array = dataset.pixel_array.astype(np.float32)
    spacing = dataset.PixelSpacing
    slice_spacing = dataset.SpacingBetweenSlices
    spacing = np.array([slice_spacing, spacing[1], spacing[0]])

    affine = np.array([0, 0, -spacing[2], 0,
                       0, -spacing[1], 0, 0,
                       -spacing[0], 0, 0, 0,
                       0, 0, 0, 1]).reshape([4, 4])

    return  array[None], affine


def read_dicom_affine(oct_path):
    dataset = pydicom.read_file(oct_path)
    dataset.SamplesPerPixel = 1
    spacing = dataset.PixelSpacing
    slice_spacing = dataset.SpacingBetweenSlices
    spacing = np.array([slice_spacing, spacing[1], spacing[0]])

    affine = np.array([0, 0, -spacing[2], 0,
                       0, -spacing[1], 0, 0,
                       -spacing[0], 0, 0, 0,
                       0, 0, 0, 1]).reshape([4, 4])

    return affine

def read_nii(filename):
    image = nib.load(filename)
    return np.array(image.get_data()), image.affine

def read_numpy(filename):
    return np.load(filename)

def read_nii_header(filename):
    return nib.load(filename)


def get_patient_dict(path, ids=None):
    scans = os.listdir(path)
    scans = [s for s in scans if os.path.isdir(os.path.join(path,s))]
    scans.sort()

    re_query = r'^(.*)_(OD|OS)'

    print(scans)

    all_scans = {}

    for s in scans:
        match = re.match(re_query, s, re.M | re.I)
        id = match.group(1)

        if ids is not None and id not in ids:
            continue

        if id not in all_scans:
            all_scans[id] = []

        all_scans[id].append(match.group(2))

    return all_scans

def generate_names(path, patient_dict):

    for idx, eyes in patient_dict.items():
        for eye in eyes:
            s = str(idx) + '_' + eye

            timestamps = os.listdir(os.path.join(path, s))

            acquisition_time = timestamps
            acquisition_time.sort(key=int)

            for t in acquisition_time:
                full_path = os.path.join(path, s, t)
                yield s, t, full_path

def read_fundus_image(path, patient, eye, time, read_annotation = True):
    data_path = os.path.join(path, patient+'_'+eye,time,'fundus.dcm')
    annotation_path = os.path.join(path, patient+'_'+eye,time,'GA_slo.npz')

    dataset = pydicom.dcmread(data_path)
    dataset.SamplesPerPixel = 1

    data = dataset.pixel_array.astype(np.float32)

    annotation = None

    if read_annotation:
        annotation = np.load(annotation_path).astype(np.uint8)


    return data, annotation

def read_oct_image(path, read_annotation = True,suffix=''):
    data_path = os.path.join(path, 'oct'+suffix+'.nii.gz')
    annotation_path = os.path.join(path, 'mask.nii.gz')

    data_header = read_nii_header(data_path)
    annotation = None

    if read_annotation:
        annotation = read_nii(annotation_path).astype(np.uint8)

    return np.array(data_header.get_data()), annotation, data_header.affine

def read_faf_image(path, read_annotation = True):
    data_path = os.path.join(path, 'faf.nii.gz')
    annotation_path = os.path.join(path, 'mask.nii.gz')

    data_header = read_nii_header(data_path)
    annotation = None

    if read_annotation:
        annotation = read_nii(annotation_path).astype(np.uint8)


    return np.array(data_header.get_data()), annotation, data_header.affine

def read_slo_image(path, read_annotation = True):
    data_path = os.path.join(path, 'slo.nii.gz')
    annotation_path = os.path.join(path, 'mask.nii.gz')

    data_header = read_nii_header(data_path)
    annotation = None

    if read_annotation:
        annotation = read_nii(annotation_path).astype(np.uint8)


    return np.array(data_header.get_data()), annotation, data_header.affine

def resize_image(image, size, order = 1):
    return ndimage.zoom(image, tuple([t/f for t,f in zip(size, image.shape)]), order=order)

def pad_to_shape(image, shape, mode='minimum'):
    diff = np.array(shape) - np.array(image.shape)

    diff[np.isnan(diff)] = 0
    diff = diff.astype(np.int32)

    pad_left = diff // 2
    pad_right = diff - pad_left

    args = {'mode':mode
    }

    if mode == 'global_minimum':
        args = {'mode': 'constant',
        'constant_values': image.min()}

    padded = np.pad(image, pad_width=tuple(zip(pad_left, pad_right)), **args)
    return padded, (pad_left, pad_right)

def revert_pad(image, padding, dim:list=None):
    pad_left, pad_right = padding

    if dim is None:
        dim = range(len(image.shape))

    start = pad_left
    end = [s - r for s, r in zip(image.shape,pad_right)]

    index = tuple([slice(s, e) if idx in dim else slice(None) for idx,(s, e) in enumerate(zip(start, end))])

    return image[index]

def revert_pad_batch(batch, output):
    pad_left, pad_right = batch['paddingimage']
    pad_left = pad_left
    pad_right = pad_right

    output = revert_pad(output, (pad_left, pad_right), dim=[1, 3])
    return output

def get_indices(position, center_shape, border):
    index = [p * c for p, c in zip(position, center_shape)]
    index_min = [i - b for i, b in zip(index, border)]
    index_max = [i + c + b for i, c, b in zip(index, center_shape, border)]

    return index_min, index_max


def copy(data, tile_shape, index_min, index_max):
    ret = torch.zeros(size=data.shape[:2]+tuple(tile_shape), dtype=torch.float)
    #ret = np.zeros(shape=data.shape[:2] + tuple(tile_shape), dtype=np.float)

    index_clamp_min = np.maximum(index_min, 0)
    index_clamp_max = np.minimum(index_max, data.shape[2:])

    diff_min = [min_c - min_i for min_c, min_i in zip(index_clamp_min, index_min)]
    diff_max = [t - (max_i - max_c) for t, max_c, max_i in zip(tile_shape, index_clamp_max, index_max)]

    # print(index_min_center, index_max_center, diff_min, diff_max)

    ret[:,:,diff_min[0]:diff_max[0], diff_min[1]:diff_max[1], diff_min[2]:diff_max[2]] = \
        data[:,:,index_clamp_min[0]:index_clamp_max[0], index_clamp_min[1]:index_clamp_max[1],
        index_clamp_min[2]:index_clamp_max[2]]

    return ret


def ravel_index(index, grid):
    i = 0
    prod = 1
    for j in reversed(range(len(grid))):
        i = i + prod * index[j]
        prod = prod * grid[j]

    return i


def unravel_index(index, grid):
    i = []
    prod = np.prod(grid)
    for j in range(len(grid)):
        prod = prod // grid[j]
        i.append(index // prod)
        index = index % prod

    return i


def copy_back(data, tile, center_shape, index_min, index_max, border):
    index_center_min = [i + b for i, b in zip(index_min, border)]
    index_center_max = [i - b for i, b in zip(index_max, border)]

    index_clamp_min = np.maximum(index_center_min, 0)
    index_clamp_max = np.minimum(index_center_max, data.shape[2:])

    diff_min = [t + min_c - min_i for t, min_c, min_i in zip(border, index_clamp_min, index_center_min)]
    diff_max = [b + t - (max_i - max_c) for b, t, max_c, max_i in
                zip(border, center_shape, index_clamp_max, index_center_max)]

    #print(index_clamp_min, index_clamp_max, diff_min, diff_max)

    data[:,:,index_clamp_min[0]:index_clamp_max[0], index_clamp_min[1]:index_clamp_max[1],
    index_clamp_min[2]:index_clamp_max[2]] = \
        tile[:,:,diff_min[0]:diff_max[0], diff_min[1]:diff_max[1], diff_min[2]:diff_max[2]]

def closest_to_k(n,k=8):
    if n % k == 0:
        return n
    else:
        return ((n // k) + 1)*k

def bbox3(img):
    """
    compute bounding box of the nonzero image pixels
    :param img: input image
    :return: bbox with shape (2,3) and contents [min,max]
    """
    rows = np.any(img, axis=1)
    rows = np.any(rows, axis=1)

    cols = np.any(img, axis=0)
    cols = np.any(cols, axis=1)

    slices = np.any(img, axis=0)
    slices = np.any(slices, axis=0)

    rows = np.where(rows)
    cols = np.where(cols)
    slices = np.where(slices)
    if (rows[0].shape[0] > 0):
        rmin, rmax = rows[0][[0, -1]]
        cmin, cmax = cols[0][[0, -1]]
        smin, smax = slices[0][[0, -1]]

        return np.array([[rmin, cmin, smin], [rmax, cmax, smax]])
    return np.array([[-1,-1,-1],[0,0,0]])

def bbox2(img):
    """
    compute bounding box of the nonzero image pixels
    :param img: input image
    :return: bbox with shape (2,2) and contents [min,max]
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)

    rows = np.where(rows)
    cols = np.where(cols)
    if (rows[0].shape[0] > 0):
        rmin, rmax = rows[0][[0, -1]]
        cmin, cmax = cols[0][[0, -1]]

        return np.array([[rmin, cmin], [rmax, cmax]])
    return np.array([[-1,-1,-1],[0,0,0]])

def get_checkpoint_path(path, metric_name, format=':.4f', start_epoch=0, n_from_best=0):
    filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    rx = re.compile('epoch=(\d+)-'+metric_name+'=(\d+\.\d+)(?:-v(?:\d+))?\.ckpt')
    results = []

    for f in filenames:
        g = rx.search(f)
        if g is None:
            continue
        r = {'epoch':int(g.group(1)),
             'val':float(g.group(2))}
        results.append(r)

    results = pd.DataFrame(results)

    results = results[results['epoch']>=start_epoch].sort_values(by='val',ascending=False)

    return ('epoch={:d}-{}={'+format+'}.ckpt').format(int(results.iloc[n_from_best,:]['epoch']),metric_name,
                                                       results.iloc[n_from_best,:]['val'])



def _array_to_cuda(array):
    if isinstance(array, torch.Tensor):
        array = array.cuda()
    elif isinstance(array, dict):
        for key in array:
            array[key] = _array_to_cuda(array[key])
    elif isinstance(array, list):
        array = [_array_to_cuda(a) for a in array]

    return array


def _array_to_cpu(array):
    if isinstance(array, torch.Tensor):
        array = array.to('cpu')
    elif isinstance(array, dict):
        for key in array:
            array[key] = _array_to_cpu(array[key])
    elif isinstance(array, list):
        array = [_array_to_cpu(a) for a in array]

    return array

def _freeze(params):
    for p in params:
        p.requires_grad = False

def _unfreeze(params):
    for p in params:
        p.requires_grad = True