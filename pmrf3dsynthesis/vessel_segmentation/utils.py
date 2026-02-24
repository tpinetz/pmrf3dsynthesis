from pmrf3dsynthesis.vessel_segmentation.model import CustomNet, ResidualBlock
from pmrf3dsynthesis.vessel_segmentation.mytransforms import PadToRate

import torch
import os
import numpy as np

def vessel_model_norm(image):
    return (image - image.mean(axis=(1,2), keepdims=True) + 1) / (image.std(axis=(1,2), keepdims=True) + 1)


def extract_vessels_oct(pat_oct, ves_model, device='cpu'):
    pat_oct_for_vessel_model = pat_oct.copy()
    pat_oct_for_vessel_model = vessel_model_norm(pat_oct_for_vessel_model * 2 ** 16)[None]
    
    record = {'image': torch.from_numpy(pat_oct_for_vessel_model.astype(np.float32))}
    transform = PadToRate(transform_keys=['image'],rate=[1,32,32,32],mode='reflect')
    test = transform(record)

    with torch.no_grad():
        vessels = ves_model({'image': torch.from_numpy(test['image'][None]).to(device)})
    return vessels['prediction'].detach().cpu().numpy()[0,0,:,0]

def load_model():
    ves_model = CustomNet(4, [1, 1, 2, 4],[4, 8, 16, 32], None, ResidualBlock)
    ckpt = torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "model.ckpt"), weights_only=False)
    ves_model.load_state_dict(ckpt)


    return ves_model