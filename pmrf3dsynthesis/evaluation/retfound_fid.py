import torch
import torch.nn as nn
from transformers import ViTModel, AutoImageProcessor
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms

class RETFoundFeatureExtractor(nn.Module):
    def __init__(self, model_name="iszt/RETFound_mae_natureOCT"):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(model_name)
        self.backbone.eval()
        
        # 1. Load the official processor config
        processor = AutoImageProcessor.from_pretrained(model_name)
        
        # 2. Extract the exact mean/std the model expects
        # The processor config usually has 'image_mean' and 'image_std'
        # If they are not present, it falls back to defaults, but this is safer.
        mean = processor.image_mean if hasattr(processor, 'image_mean') else [0.5, 0.5, 0.5]
        std = processor.image_std if hasattr(processor, 'image_std') else [0.5, 0.5, 0.5]
        
        # 3. Extract target size (usually {"height": 224, "width": 224})
        # Some processors use 'size', others 'crop_size'
        if hasattr(processor, 'size') and isinstance(processor.size, dict):
            h, w = processor.size['height'], processor.size['width']
        elif hasattr(processor, 'crop_size') and isinstance(processor.crop_size, dict):
             h, w = processor.crop_size['height'], processor.crop_size['width']
        else:
            h, w = 224, 224 # Fallback
            
        print(f"Loaded Preprocessing -- Mean: {mean}, Std: {std}, Size: {h}x{w}")

        self.transform = transforms.Compose([
            transforms.Resize((h, w), antialias=True),
            transforms.Normalize(mean=mean, std=std)
        ])

    def forward(self, x):
        # Input x is expected to be [0, 1] float tensors
        
        # 1. Handle Grayscale -> RGB
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # 2. Apply Transforms (Resize + Normalize)
        x = self.transform(x)
        
        # 3. Forward pass
        with torch.no_grad():
            outputs = self.backbone(pixel_values=x)
            
        # 4. Extract CLS token
        return outputs.last_hidden_state[:, 0, :]

# --- Main Usage ---

if __name__ == "__main__":

    # 1. Initialize FID with the Custom Wrapper
    # We set normalize=True because we will pass Float tensors in [0, 1] range.
    # If you pass uint8 [0, 255], set normalize=False.
    custom_fid = FrechetInceptionDistance(
        feature=RETFoundFeatureExtractor(), 
        normalize=True  
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    custom_fid.to(device)

    # 2. Simulate Data (Grayscale, Float 0-1)
    # 100 images, 1 channel, 256x256
    real_imgs = torch.rand(100, 1, 256, 256).to(device)
    fake_imgs = torch.rand(100, 1, 256, 256).to(device)

    # 3. Update Metric
    print("Updating Real stats...")
    # torchmetrics automatically batches this if you pass a large tensor, 
    # or you can loop over your dataloader.
    custom_fid.update(real_imgs, real=True)

    print("Updating Fake stats...")
    custom_fid.update(fake_imgs, real=False)

    # 4. Compute
    fid_score = custom_fid.compute()
    print(f"RETFound-FID Score: {fid_score.item():.4f}")