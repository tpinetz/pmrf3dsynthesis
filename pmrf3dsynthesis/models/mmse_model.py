import torch
from typing import Dict
import lightning

from omegaconf import OmegaConf
from torchvision.utils import make_grid

from pmrf3dsynthesis.models.losses.discriminator import NLayerDiscriminator
from pmrf3dsynthesis.models.losses.ganloss import GANLoss

from pmrf3dsynthesis.data import constants as C
from pmrf3dsynthesis.models.edm_network import Song3DUNet


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
            self.init_projection = torch.nn.Conv2d(1, config.get("slices", 64), kernel_size=1, stride=1, padding=0)
            self.network = Song3DUNet(config.get("image_resolution", 224), config.get("in_channels", 1),
            config.get("out_channels", 1), model_channels=config.get("model_channels", 64))
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

        if self.config.get("use_l1_loss", False):
            self.criterion = torch.nn.L1Loss()
        else:
            self.criterion = torch.nn.MSELoss()


    def configure_optimizers(self):
        if self.use_pixtopix:
            g_opt = torch.optim.Adam(list(self.network.parameters()) + list(self.init_projection.parameters()), lr=self.config.get("lr_gen", 2e-4))
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
        return self.network(self.init_projection(x).permute(0, 2, 1, 3)[:,None])

    def predict(self, x):
        return self.network.predict(self.init_projection(x).permute(0, 2, 1, 3)[:,None])

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def to_2dinput(self, x):
        b, c, d, h, w = x.shape # c should be 1 for grayscale
        x_2d = x.view(b*d, 1, h, w)
        return x_2d

    def training_step(self, batch, batch_idx):
        x, y = batch[C.IMAGE], batch[C.TARGET]

        if self.network_name == C.MODEL_TRANSUPNET:
            y_hat = self.network(x)
            gt = y.clone()
            loss = 0
            for i in range(len(y_hat) - 1, -1, -1):
                loss += self.criterion(y_hat[i], gt)
                if i > 0:
                    gt = torch.nn.functional.interpolate(gt, size=y_hat[i - 1].shape[2:], mode='trilinear', align_corners=True)
            y_hat = y_hat[-1]
        else:
            x_3d = self.init_projection(x).permute(0, 2, 1, 3)[:,None]
            y_hat = self.network(x_3d)
            loss = self.criterion(y_hat, y)
        if self.config.get("use_pixtopix", False):
            g_opt, d_opt = self.optimizers()

            if batch_idx % 2 == 0:
                gan_loss = self.pix2pixLoss(self.DomainDisc(self.to_2dinput(y_hat)), True)
                self.log("loss/gan_loss_fake", gan_loss.detach())
                loss += gan_loss
                g_opt.zero_grad()
                self.manual_backward(loss)
                g_opt.step()
            else:
                disc_loss = (self.pix2pixLoss(self.DomainDisc(self.to_2dinput(y_hat.detach())), False) +
                             self.pix2pixLoss(self.DomainDisc(self.to_2dinput(y)), True))
                self.log("loss/disc_loss", disc_loss.detach())
                d_opt.zero_grad()
                self.manual_backward(disc_loss)
                d_opt.step() 
            

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

            if self.network_name == C.MODEL_TRANSUPNET:
                y_hat = self.network(x)
                y_hat = y_hat[-1]
            else:
                x_3d = self.init_projection(x).permute(0, 2, 1, 3)[:,None]
                y_hat = self.network(x_3d)

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

if __name__ == "__main__":
    model = MMSEModel(network_name=C.SONG_3DUNET)

    x = torch.randn(2, 1, 64, 224, 224)
    with torch.no_grad():
        print(model(x).shape)
