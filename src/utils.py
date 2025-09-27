import torch
from PIL import Image
import numpy as np

def crop_to_multiple_of_16(img):
    width, height = img.size
    
    # Calculate new dimensions that are multiples of 8
    new_width = width - (width % 16)  
    new_height = height - (height % 16)
    
    # Calculate crop box coordinates
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    
    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))
    
    return cropped_img

def resize_and_pad_to_size(image, target_width, target_height):
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    # Get original dimensions
    orig_width, orig_height = image.size
    
    # Calculate aspect ratios
    target_ratio = target_width / target_height
    orig_ratio = orig_width / orig_height
    
    # Calculate new dimensions while maintaining aspect ratio
    if orig_ratio > target_ratio:
        # Image is wider than target ratio - scale by width
        new_width = target_width
        new_height = int(new_width / orig_ratio)
    else:
        # Image is taller than target ratio - scale by height
        new_height = target_height
        new_width = int(new_height * orig_ratio)
        
    # Resize image
    resized_image = image.resize((new_width, new_height))
    
    # Create white background image of target size
    padded_image = Image.new('RGB', (target_width, target_height), 'white')
    
    # Calculate padding to center the image
    left_padding = (target_width - new_width) // 2
    top_padding = (target_height - new_height) // 2
    
    # Paste resized image onto padded background
    padded_image.paste(resized_image, (left_padding, top_padding))
    
    return padded_image, left_padding, top_padding, target_width - new_width - left_padding, target_height - new_height - top_padding

def resize_by_height(image, height):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # image is a PIL image
    image = image.resize((int(image.width * height / image.height), height))
    return crop_to_multiple_of_16(image)

# Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._pack_latents    
def flux_pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

# Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._unpack_latents
def flux_unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents

# TODO: it is more reasonable to have target pe staring at 0
def prepare_latent_image_ids(height, width_tgt, height_spa, width_spa, height_sub, width_sub, device, dtype):
    # assert width_spa==0 or width_tgt==width_spa
    latent_image_ids = torch.zeros(height, width_tgt, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height, device=device)[:, None]  # y坐标
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width_tgt, device=device)[None, :]   # x坐标
    
    cond_mark = 0
    if width_spa>0:
        cond_mark += 1
        condspa_image_ids = torch.zeros(height_spa, width_spa, 3, device=device, dtype=dtype)
        condspa_image_ids[..., 0] = cond_mark
        condspa_image_ids[..., 1] = condspa_image_ids[..., 1] + torch.arange(height_spa, device=device)[:, None]
        condspa_image_ids[..., 2] = condspa_image_ids[..., 2] + torch.arange(width_spa, device=device)[None, :]
        condspa_image_ids = condspa_image_ids.reshape(-1, condspa_image_ids.shape[-1])
        

    if width_sub>0:
        cond_mark += 1
        condsub_image_ids = torch.zeros(height_sub, width_sub, 3, device=device, dtype=dtype)
        condsub_image_ids[..., 0] = cond_mark
        condsub_image_ids[..., 1] = condsub_image_ids[..., 1] + torch.arange(height_sub, device=device)[:, None]
        condsub_image_ids[..., 2] = condsub_image_ids[..., 2] + torch.arange(width_sub, device=device)[None, :] + width_tgt
        condsub_image_ids = condsub_image_ids.reshape(-1, condsub_image_ids.shape[-1])

    latent_image_ids = latent_image_ids.reshape(-1, latent_image_ids.shape[-1])
    latent_image_ids = torch.cat([latent_image_ids, condspa_image_ids],dim=-2) if width_spa>0 else latent_image_ids
    latent_image_ids = torch.cat([latent_image_ids, condsub_image_ids],dim=-2) if width_sub>0 else latent_image_ids
    return latent_image_ids