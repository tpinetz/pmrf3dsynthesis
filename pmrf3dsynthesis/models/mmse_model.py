import torch
from typing import Dict
import lightning
from omegaconf import OmegaConf
from torchvision.utils import make_grid

from pmrf3dsynthesis.models.losses.discriminator import NLayerDiscriminator
from pmrf3dsynthesis.models.losses.ganloss import GANLoss

from pmrf3dsynthesis.data import constants as C
from pmrf3dsynthesis.models.edm_network import Song3DUNet
from pmrf3dsynthesis.models.transupnet import TransUpNet

import logging

ch = logging.StreamHandler()
logging.basicConfig(level=logging.INFO,
                    handlers=[
                        ch,
                    ])

class MMSEModel(lightning.LightningModule):

    def __init__(self, **config: Dict):
        super().__init__()
        self.config = OmegaConf.create(config)
        self.save_hyperparameters()
        
        logging.info(f"Model config: {config}")


        self.network_name = config.get('network_name', C.MODEL_UNETPLUSPLUS)

        if self.network_name == C.SONG_3DUNET:
            self.ct_transform = config.get("ct_transform", False)
            self.init_projection = torch.nn.Conv2d(config.get("in_channels", 1), config.get("slices", 64), kernel_size=1, stride=1, padding=0)
            self.network = Song3DUNet(config.get("image_resolution", 224), 1,
            config.get("out_channels", 1), model_channels=config.get("model_channels", 64))
        elif self.network_name == C.MODEL_TRANSUPNET:
            from .transupnet import TransUpNet
            self.network = TransUpNet(in_channels=config.get("in_channels", 1),
                                      slides=config.get("slices", 256), ct_transform=config.get("ct_transform", False),
                                      residual_prediction=config.get("residual_prediction", True))
        else:
            raise NotImplementedError()
        
        self.validation_metrics = {
            "recon": []
        }
        self.validation_step_outputs = []
        self.validation_step_segmentation_outputs = []
        self.use_pixtopix = config.get("use_pixtopix", False)
        if self.use_pixtopix:
            self.DomainDisc = NLayerDiscriminator(config.get("domain_channels", 1))
            self.pix2pixLoss = GANLoss(gan_mode="lsgan")
            self.automatic_optimization = False
            self.ganlossweight = self.config.get("ganlossweight", 0.01)

        if self.config.get("use_l1_loss", False):
            self.criterion = torch.nn.L1Loss()
        elif self.config.get("use_wloss", False):
            self.criterion = lambda x, y: PatchWiseWasserSteinSinkhorn(x, y, 4, 4)
        else:
            self.criterion = torch.nn.MSELoss()


    def configure_optimizers(self):
        if self.use_pixtopix:
            g_parameters = list(self.network.parameters())
            if self.network_name != C.MODEL_TRANSUPNET:
                g_parameters += list(self.init_projection.parameters())
            g_opt = torch.optim.Adam(g_parameters, lr=self.config.get("lr_gen", 2e-4))
            d_opt = torch.optim.Adam(list(self.DomainDisc.parameters()), lr=self.config.get("lr_disc", 2e-4))
            return [g_opt, d_opt]

        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.learning_rate
        )
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                eta_min=1e-6,
                T_max=self.config.num_iter
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        if self.network_name == C.MODEL_TRANSUPNET:
            return self.network(x)
        else:
            if self.ct_transform:
                x_3d = self.init_projection(x)[:, None]
            else:
                x_3d = self.init_projection(x).permute(0, 2, 1, 3)[:,None]
            return self.network(x_3d)
    
    @torch.no_grad()
    def predict(self, x):
        self.eval()
        y_hat = self.forward(x)
        if isinstance(y_hat, list):
            y_hat = y_hat[-1]
        return y_hat

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def to_2dinput(self, x):
        b, c, d, h, w = x.shape # c should be 1 for grayscale
        x_2d = x.view(b*d, 1, h, w)
        return x_2d

    def training_step(self, batch, batch_idx):
        x, y = batch[C.IMAGE], batch[C.TARGET]
        y_hat = self.forward(x)

        if self.network_name == C.MODEL_TRANSUPNET:
            y_hat_list = y_hat
            y_hat = y_hat[-1]
        
        if self.config.get("use_pixtopix", False):
            g_opt, d_opt = self.optimizers()

            if batch_idx % 2 == 0:
                if self.network_name == C.MODEL_TRANSUPNET:
                    gt = y.clone()
                    loss = 0
                    for i in range(len(y_hat_list) - 1, -1, -1):
                        loss += self.criterion(y_hat_list[i], gt)
                        if i > 0:
                            gt = torch.nn.functional.interpolate(gt, size=y_hat_list[i - 1].shape[2:], mode='trilinear', align_corners=True)
                else:
                    loss = self.criterion(y_hat, y)
                gan_loss = self.pix2pixLoss(self.DomainDisc(self.to_2dinput(y_hat)), True)
                self.log("loss/gan_loss_fake", gan_loss.detach())
                loss += self.ganlossweight * gan_loss
                g_opt.zero_grad()
                self.manual_backward(loss)
                g_opt.step()
            else:
                loss = (self.pix2pixLoss(self.DomainDisc(self.to_2dinput(y_hat.detach())), False) +
                             self.pix2pixLoss(self.DomainDisc(self.to_2dinput(y)), True))
                self.log("loss/disc_loss", loss.detach())
                d_opt.zero_grad()
                self.manual_backward(loss)
                d_opt.step()
        else:
            if self.network_name == C.MODEL_TRANSUPNET:
                gt = y.clone()
                loss = 0
                for i in range(len(y_hat_list) - 1, -1, -1):
                    loss += self.criterion(y_hat_list[i], gt)
                    if i > 0:
                        gt = torch.nn.functional.interpolate(gt, size=y_hat_list[i - 1].shape[2:], mode='trilinear', align_corners=True)
            else:
                loss = self.criterion(y_hat, y)

        self.log("loss/train", loss.detach())

        step = self.global_step
        with torch.no_grad():
            if step % self.config.num_log == 0 or step == self.config.num_iter - 1:
                self.logger.experiment.add_image(
                    f'train-OCT',
                    make_grid(
                        torch.concat([
                        y_hat[:, 0:1, 32, :].detach(),
                        y[:, 0:1,  32, :].detach()
                        ], dim=-1),
                        nrow=4, normalize=True
                    ),
                    global_step=step
                )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[C.IMAGE], batch[C.TARGET]
        
        with torch.no_grad():
            y_hat = self.forward(x)

            if self.network_name == C.MODEL_TRANSUPNET:
                y_hat = y_hat[-1]

            loss = self.criterion(y_hat, y)

            self.validation_metrics["recon"].append(
                    loss
                )

            if batch_idx < 10:
                self.validation_step_outputs.append(
                    torch.concat([
                        y_hat[0, 0:1, 32, :],
                        y[0, 0:1, 32, :]
                    ], dim=-1)
                )

    def on_validation_epoch_end(self):
        for k, v in self.validation_metrics.items():
            self.log(f"{k}/val", sum(v) / len(v), sync_dist=True)
            v.clear()

        diffs = torch.stack(self.validation_step_outputs, dim=0)
        self.logger.experiment.add_image(
            'val-OCT',
            make_grid(diffs, nrow=2, normalize=True),
            global_step=self.global_step
        )
        self.validation_step_outputs.clear()
