import torch
import numpy as np
from PIL import Image
import argparse
import os

from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel, CLIPTextModel
from diffusers import FluxInpaintPipeline, AutoencoderKL
from diffusers.hooks import apply_group_offloading
from src.pipeline_tryon import FluxTryonPipeline
from src.utils import crop_to_multiple_of_16, resize_and_pad_to_size, resize_by_height

def load_models(model_path, lora_name=None, device="cuda", torch_dtype=torch.bfloat16, group_offloading=False):
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch_dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder_2", torch_dtype=torch_dtype)
    transformer = FluxTransformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch_dtype)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch_dtype)

    pipe = FluxTryonPipeline.from_pretrained(
        model_path,
        transformer=transformer,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        vae=vae,
        torch_dtype=torch_dtype,
    )
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    if lora_name is not None:
        pipe.enable_model_cpu_offload()
        pipe.load_lora_weights(
            "loooooong/Any2anyTryon",
            weight_name=lora_name,
            adapter_name="tryon",
        )
        pipe.remove_all_hooks()
    
    if group_offloading:
        # https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux#group-offloading
        apply_group_offloading(
            pipe.transformer,
            offload_type="leaf_level",
            offload_device=torch.device("cpu"),
            onload_device=torch.device(device),
            use_stream=True,
        )
        apply_group_offloading(
            pipe.text_encoder, 
            offload_device=torch.device("cpu"),
            onload_device=torch.device(device),
            offload_type="leaf_level",
            use_stream=True,
        )
        apply_group_offloading(
            pipe.text_encoder_2, 
            offload_device=torch.device("cpu"),
            onload_device=torch.device(device),
            offload_type="leaf_level",
            use_stream=True,
        )
        apply_group_offloading(
            pipe.vae, 
            offload_device=torch.device("cpu"),
            onload_device=torch.device(device),
            offload_type="leaf_level",
            use_stream=True,
        )
    pipe.to(device=device)
    return pipe

@torch.no_grad()
def generate_image(pipe, model_image_path, garment_image_path, prompt="", height=512, width=384, 
                  seed=0, guidance_scale=3.5, num_inference_steps=30):
    height, width = int(height), int(width)
    width = width - (width % 16)  
    height = height - (height % 16)

    concat_image_list = [Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))]
    has_model_image = model_image_path is not None
    has_garment_image = garment_image_path is not None

    if has_model_image:
        model_image = Image.open(model_image_path)
        if has_garment_image:
            input_height, input_width = model_image.size[1], model_image.size[0]
            model_image, lp, tp, rp, bp = resize_and_pad_to_size(model_image, width, height)
        else:
            model_image = resize_by_height(model_image, height)
        concat_image_list.append(model_image)

    if has_garment_image:
        garment_image = Image.open(garment_image_path)
        garment_image = resize_by_height(garment_image, height)
        concat_image_list.append(garment_image)

    image = Image.fromarray(np.concatenate([np.array(img) for img in concat_image_list], axis=1))
    
    mask = np.zeros_like(np.array(image))
    mask[:,:width] = 255
    mask_image = Image.fromarray(mask)
    
    image = pipe(
        prompt,
        image=image,
        mask_image=mask_image,
        strength=1.,
        height=height,
        width=image.width,
        target_width=width,
        tryon=has_model_image and has_garment_image,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="pil",
    ).images[0]
    
    if has_model_image and has_garment_image:
        image = image.crop((lp, tp, image.width-rp, image.height-bp)).resize((input_width, input_height))
    
    return image

def main():
    parser = argparse.ArgumentParser(description='Virtual Try-on Image Generation')
    parser.add_argument('--model_path', type=str, default="black-forest-labs/FLUX.1-dev", help='Path to the model')
    parser.add_argument('--lora_name', type=str, default="dev_lora_any2any_alltasks.safetensors", help='choose from dev_lora_any2any_multi.safetensors, dev_lora_any2any_alltasks.safetensors, dev_lora_any2any_tryon.safetensors and dev_lora_garment_reconstruction.safetensors')
    parser.add_argument('--model_image', type=str, help='Path to the model image')
    parser.add_argument('--garment_image', type=str, help='Path to the garment image')
    parser.add_argument('--prompt', type=str, default="")
    parser.add_argument('--height', type=int, default=576)
    parser.add_argument('--width', type=int, default=576)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--guidance_scale', type=float, default=3.5)
    parser.add_argument('--num_inference_steps', type=int, default=30)
    parser.add_argument('--output_path', type=str, default='./results/output.png')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--group_offloading', action="store_true")
    
    args = parser.parse_args()
    
    pipe = load_models(args.model_path, lora_name=args.lora_name, device=args.device,group_offloading=args.group_offloading)
    
    output_image = generate_image(
        pipe=pipe,
        model_image_path=args.model_image,
        garment_image_path=args.garment_image,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps
    )
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    output_image.save(args.output_path)
    print(f"Generated image saved to {args.output_path}")

if __name__ == "__main__":
    main()