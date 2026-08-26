import numpy as np
import torch
from typing import Dict
import lightning
from omegaconf import OmegaConf

from torchvision.utils import make_grid
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from pmrf3dsynthesis.models.edm_network import SongUNet, Song3DUNet
from pmrf3dsynthesis.models.model_jit import JiT_models
from pmrf3dsynthesis.data import constants as C

import logging

ch = logging.StreamHandler()
logging.basicConfig(level=logging.INFO,
                    handlers=[
                        ch,
                    ])



class FlowModel(lightning.LightningModule):

    def __init__(self, **config: Dict):
        super().__init__()
        self.config = OmegaConf.create(config)
        self.save_hyperparameters()

        self.mmse_model = None
        logging.info(f"Model config: {config}")
        self.sigma_data = config.get("sigma_data", 0.5)
        self.sigma_start = config.get("sigma_start", 0.0)
        self.network_name = config.get("network_name", C.SONG_UNET)
        self.t_eps = config.get("t_eps", 5e-2)
        self.map_2d_to_3d = config.get("map_2d_to_3d", False)
        self.ct_transform = config.get("ct_transform", False)

        if self.network_name == C.SONG_UNET:
            self.network = SongUNet(config.get("image_resolution", 224), config.get("in_channels", 1), config.get("out_channels", 1), model_channels=config.get("model_channels", 64))
        elif self.network_name == C.JIT:
            self.network = JiT_models[config.get('jit_model', 'JiT-B/16')](input_size=config.get("image_resolution", 224))
        elif self.network_name == C.SONG_3DUNET:
            if self.map_2d_to_3d:
                self.init_projection = torch.nn.Conv2d(1 if self.ct_transform else 3, config.get("slices", 64), kernel_size=1, stride=1, padding=0)
            
            self.network_3d = Song3DUNet(config.get("image_resolution", 224), config.get("in_channels", 1),
            config.get("out_channels", 1), model_channels=config.get("model_channels", 16))
            self.network = lambda x, t, c: self.network_3d(x)       # my implementation of Song3DUNet does not use time or classes
        else:
            raise NotImplementedError(f"Network {self.network_name} not implemented in FlowModel!")

        self.validation_metrics = {
            "recon": []
        }
        self.validation_step_outputs = []
        self.validation_step_segmentation_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate
        )
        scheduler = {
            'scheduler': LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.config.get('warmup_epochs', 0),
                max_epochs=-1  # to be set by the trainer
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def set_mmse_model(self, mmse_model):
        self.mmse_model = mmse_model

    def forward(self, x, T, noise_type=C.NOISE_TYPE_NORMAL):
        with torch.no_grad():
            if self.map_2d_to_3d:
                x = self.init_projection(x).permute(0, 2, 1, 3)[:,None]
            if self.mmse_model is not None:
                x = self.mmse_model.predict(x)

                if noise_type == C.NOISE_TYPE_NORMAL:
                    noise = torch.randn_like(x)
                elif noise_type == C.NOISE_TYPE_CONSTANT:
                    noise = torch.randn_like(x)[:,:,0:1,:,:].expand(-1,-1,x.shape[2],-1,-1)
                elif noise_type == C.NOISE_TYPE_SLERP:
                    eps = 1e-8
                    z0 = torch.randn_like(x[:,0,0])
                    z1 = torch.randn_like(x[:,0,0])
                    alphas = torch.linspace(0, 1, x.shape[2]).to(x.device)

                    z0_flat = z0.reshape(-1)
                    z1_flat = z1.reshape(-1)
                    omega = torch.acos(torch.clamp(torch.dot(z0_flat, z1_flat) / (z0_flat.norm() * z1_flat.norm() + eps), -1.0 + 1e-7, 1.0 - 1e-7)) 
                    sin_omega = torch.sin(omega)
                    w1 = torch.sin((1 - alphas) * omega) / (sin_omega + eps)
                    w2 = torch.sin(alphas * omega) / (sin_omega + eps)
                    noise = (w1.unsqueeze(-1).unsqueeze(-1).unsqueeze(0) * z0 +
                             w2.unsqueeze(-1).unsqueeze(-1).unsqueeze(0) * z1).unsqueeze(1)

                x_orig = x + self.sigma_start * noise
                y_hat = x_orig.clone()
                for _slice in range(x.shape[2]):
                    for i in range(T):
                        test = self.network(y_hat[:, :, _slice, :, :], i / T * torch.ones((1), dtype=torch.float32).to(y_hat.device), torch.zeros((1), dtype=torch.int).to(y_hat.device))
                        vel = (test - y_hat[:,:, _slice]) /  max((1.0 - i / T), self.t_eps)  
                        y_hat[:,:,_slice] += 1 / T * vel

            else:
                logging.info("No MMSE model set for FlowModel, passing input directly to flow model.")
                x_orig = x + self.sigma_start * torch.randn_like(x)
                y_hat = x_orig.clone()
                for i in range(T):
                    test = self.network(y_hat, i / T * torch.ones((1), dtype=torch.float32).to(y_hat.device), torch.zeros((1), dtype=torch.int).to(y_hat.device))
                    vel = (test - y_hat) /  max((1.0 - i / T), self.t_eps)  
                    y_hat += 1 / T * vel
        
        return y_hat

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def extract_img(self, tensor):
        if tensor.ndim == 4:
            return torch.clamp(tensor[:, 0:1, :, :].detach(), -1., 1.)
        elif tensor.ndim == 5:
            B, C, D, H, W = tensor.shape
            return torch.clamp(tensor[:, 0:1, D // 2, :, :].detach(), -1., 1.)
        else:
            raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        x, y = batch[C.IMAGE], batch[C.TARGET]
        if self.map_2d_to_3d:
            if self.ct_transform:
                x = self.init_projection(x)[:,None]
            else:
                x = self.init_projection(x).permute(0, 2, 1, 3)[:,None]
        if x.ndim == 4:
            t = torch.rand(x.shape[0], device=x.device).view(-1, 1, 1, 1)
        elif x.ndim == 5:
            t = torch.rand(x.shape[0], device=x.device).view(-1, 1, 1, 1, 1)
        else:
            raise NotImplementedError()

        n = torch.randn_like(x, device=x.device) * self.sigma_start
        
        xt = t * y + (1 - t) * (x + n)
        v = (y - xt) / (1 - t).clamp_min(self.t_eps)

        y_hat = self.network(xt, t.flatten(), torch.zeros(t.shape[0], device=t.device, dtype=torch.int))

        v_pred = (y_hat - xt) / (1 - t).clamp_min(self.t_eps)
        loss = ((v - v_pred) ** 2).mean(axis=(1,2,3)).mean()
        # loss = ((y_hat - y) ** 2).mean(axis=(1,2,3)).mean()

        self.log("loss/train", loss.item())

        step = self.global_step
        with torch.no_grad():
            if step % self.config.num_log == 0 or step == self.config.num_iter - 1:
                self.logger.experiment.add_image(
                    f'train-OCT',
                    make_grid(
                        torch.concat([
                        self.extract_img(x),
                        self.extract_img(xt),
                        self.extract_img(y_hat),
                        self.extract_img(y)
                        ], dim=-1),
                        nrow=4, normalize=True
                    ),
                    global_step=step
                )
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch[C.IMAGE], batch[C.TARGET]


        if self.map_2d_to_3d and x.ndim == 4:
            if self.ct_transform:
                x = self.init_projection(x)[:,None]
            else:
                x = self.init_projection(x).permute(0, 2, 1, 3)[:,None]

        if batch_idx < 10:
            with torch.no_grad():
                y_hat = x + torch.randn_like(x, device=x.device) * self.sigma_data
                T = 20
                for i in range(T):
                    y_c = self.network(y_hat,
                                    i / T * torch.ones(x.shape[0], device=x.device), torch.zeros(x.shape[0], device=x.device, dtype=torch.int))
                    v = (y_c - y_hat) / max((1 - i / T), self.t_eps)
                    # Euler step
                    y_hat += 1 / T * v 
                
                loss = (((y_hat - y) ** 2)).mean()

                self.validation_metrics["recon"].append(
                        loss
                    )

                self.validation_step_outputs.append(
                    torch.concat([
                        self.extract_img(x),
                        self.extract_img(y_hat),
                        self.extract_img(y)
                    ], dim=-1)
                )

    def on_validation_epoch_end(self):
        for k, v in self.validation_metrics.items():
            self.log(f"{k}/val", sum(v) / len(v), sync_dist=True)
            v.clear()

        diffs = torch.concat(self.validation_step_outputs, dim=0)
        self.logger.experiment.add_image(
            'val-OCT',
            make_grid(diffs, nrow=2, normalize=True),
            global_step=self.global_step
        )
        self.validation_step_outputs.clear()
