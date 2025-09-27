from diffusers import FluxTransformer2DModel
from transformers import T5EncoderModel, CLIPTextModel
from diffusers import AutoencoderKL
from src.pipeline_tryon import FluxTryonPipeline

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime
from pathlib import Path
import argparse


torch_dtype = torch.bfloat16
model_caption_data, garment_caption_data, model_caption_inpaint = {},{},{}
model_caption_json = "data/zalando-hd-resized/model_caption.json"
if os.path.exists(model_caption_json):
    with open(model_caption_json, "r") as f:
        model_caption_data = json.load(f)
garment_caption_json = "data/zalando-hd-resized/cloth_caption.json"
if os.path.exists(garment_caption_json):
    with open(garment_caption_json, "r") as f:
        garment_caption_data = json.load(f)
model_inpaint_caption_json = "data/zalando-hd-resized/model_inpaint_caption.json"
if os.path.exists(model_inpaint_caption_json):
    with open(model_inpaint_caption_json, "r") as f:
        model_caption_inpaint = json.load(f)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="data/zalando-hd-resized/test/image")
    parser.add_argument('--garment_dir', type=str, default="data/zalando-hd-resized/test/cloth")
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--meta_file', type=str, default="data/zalando-hd-resized/test_pairs.txt")
    parser.add_argument('--model_path', type=str, default="black-forest-labs/FLUX.1-dev", help='Path to the model')
    parser.add_argument("--train_double_block_only", action="store_true")
    parser.add_argument('--paired', action="store_true")
    parser.add_argument('--repaint', action="store_true")
    parser.add_argument("--mask_dir", type=str, default=None, help="Directory containing mask images")
    parser.add_argument("--source_dir", type=str, default=None, help="Directory containing source model images") 
    return parser.parse_args()

def load_models(model_path, device="cuda"):
    # Enable memory efficient attention
    text_encoder = CLIPTextModel.from_pretrained(f"{model_path}/text_encoder", torch_dtype=torch_dtype,)
    text_encoder_2 = T5EncoderModel.from_pretrained(f"{model_path}/text_encoder_2", torch_dtype=torch_dtype,)
    transformer = FluxTransformer2DModel.from_pretrained(f"{model_path}/transformer", torch_dtype=torch_dtype,)
    vae = AutoencoderKL.from_pretrained(f"{model_path}/vae")
    
    pipe = FluxTryonPipeline.from_pretrained(
        model_path,
        transformer=transformer,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        vae=vae,
        torch_dtype=torch_dtype,
    ).to(device=device, dtype=torch_dtype)
    
    # Enable memory efficient attention and VAE optimization
    pipe.enable_attention_slicing()
    # pipe.enable_sequential_cpu_offload()
    # pipe.enable_model_cpu_offload()
    vae.enable_slicing()
    vae.enable_tiling()
    pipe.load_lora_weights(
        "loooooong/Any2anyTryon",
        weight_name="dev_lora_tryon_vitonhd_512.safetensors",
        adapter_name="tryon",
    )

    from diffusers.utils.torch_utils import randn_tensor
    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype, target_width=-1, edit=False):
        latent_image_ids = torch.zeros(height, width, 3)
        if target_width==-1:
            latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
            latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
        else:
            latent_image_ids[:, :-target_width, 0] = 1
            # height keep as before
            latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
            if edit:
                latent_image_ids[:, target_width:-target_width, 0] = 2
                # left
                latent_image_ids[:, :-target_width, 2] = latent_image_ids[:, :-target_width, 2] + torch.arange(width-target_width)[None, :]
                # right
                latent_image_ids[:, -target_width:, 2] = latent_image_ids[:, -target_width:, 2] + torch.arange(target_width)[None, :]
            else:
                latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]                
        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        return latent_image_ids.to(device=device, dtype=dtype)
    
    def prepare_latents(
        self,
        image,
        timestep,
        batch_size,
        num_channels_latents,
        height,
        width,
        target_width,
        edit,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)
        sp = 2 * (int(target_width) // (self.vae_scale_factor * 2))//2 # -1 
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype, sp, edit)

        image = image.to(device=device, dtype=dtype)
        # image_latents = self._encode_vae_image(image=image, generator=generator)
        img_parts = [image[:,:,:,:-target_width], image[:,:,:,-target_width:]]
        image_latents = [self._encode_vae_image(image=img, generator=generator) for img in img_parts]
        image_latents = torch.cat(image_latents, dim=-1)

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self.scheduler.scale_noise(image_latents, timestep, noise)
        else:
            noise = latents.to(device)
            latents = noise

        noise = self._pack_latents(noise, batch_size, num_channels_latents, height, width)
        image_latents = self._pack_latents(image_latents, batch_size, num_channels_latents, height, width)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        return latents, noise, image_latents, latent_image_ids
    pipe._prepare_latent_image_ids = _prepare_latent_image_ids
    pipe.prepare_latents = prepare_latents.__get__(pipe)

    return pipe

def resize_by_height(image, height):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # image is a PIL image
    image = image.resize((int(image.width * height / image.height), height))
    width = image.width - (image.width % 16)
    height = image.height - (image.height % 16)
    return image.resize((width, height))

from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

@torch.no_grad
def generate_image(pipe, prompt, model_image, garment_image, height=512, width=384, seed=0, guidance_scale=3.5, train_double_block_only=False):
    height, width = int(height), int(width)
    width = width - (width % 16)  
    height = height - (height % 16)

    concat_image_list = []
    model_image = resize_by_height(model_image, height)
    garment_image = resize_by_height(garment_image, height)
    concat_image_list.extend([model_image, garment_image,Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))])

    image = np.concatenate([np.array(img) for img in concat_image_list], axis=1)
    image = Image.fromarray(image)
    
    mask = np.zeros_like(np.array(image))
    mask[:,-width:] = 255
    mask_image = Image.fromarray(mask)

    def forward_flux(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        img_ids_old = pipe._prepare_latent_image_ids(1, int(image.height) // (pipe.vae_scale_factor * 2), int(image.width) // (pipe.vae_scale_factor * 2), pipe.device, torch_dtype),
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ):
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                print(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            print(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            print(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)
        image_rotary_emb2 = self.pos_embed(torch.cat((txt_ids, img_ids_old), dim=0))

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs = {"use_reentrant": False}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
        
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        image_rotary_emb = image_rotary_emb2

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs = {"use_reentrant": False}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
    if train_double_block_only:
        pipe.transformer.forward = forward_flux.__get__(pipe.transformer)
    output = pipe(
        prompt,
        image=image,
        mask_image=mask_image,
        strength=1.,
        height=height,
        width=image.width,
        target_width=width,
        edit=True,
        guidance_scale=guidance_scale,
        num_inference_steps=30,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="latent",
    ).images

    latents = pipe._unpack_latents(output, image.height, image.width, pipe.vae_scale_factor)
    latents = latents[:,:,:,-width//pipe.vae_scale_factor:]
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    
    return image

def inference(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    pairs_dict = {}
    with open(args.meta_file, "r") as f:
        pairs = f.read().splitlines()
    for line in pairs:
        img_fn, gmt_fn = line.split()
        pairs_dict[img_fn] = img_fn if args.paired else gmt_fn

    pipe = load_models(args.model_path, device=device)
    
    scale = 1.0
    pipe.set_adapters("tryon",adapter_weights=[scale]) 
    height, width = 512, 384 # 768, 576 # 1024, 768 # 
    if args.output_dir is None:
        args.output_dir = f"./results/vitonhd_test_{('paired' if args.paired else 'unpaired')}-"+f"_{height}_{width}"

    os.makedirs(args.output_dir, exist_ok=True)
    
    model_files = sorted(os.listdir(args.model_dir)) # sorted(list(pairs_dict.keys())) #
    
    # Split work across GPUs
    model_files = model_files[rank::world_size]
    
    for model_file in model_files:
        name = Path(model_file).stem
        garment_file = pairs_dict[model_file]
        
        if not os.path.exists(os.path.join(args.garment_dir, garment_file)):
            print(f"Skipping {model_file} - no matching garment image")
            continue
            
        output_path = os.path.join(args.output_dir, f"{name}.jpg")
        if os.path.exists(output_path):
            print(f"Skipping {model_file} - output already exists")
            continue
            
        model_image = Image.open(os.path.join(args.model_dir, model_file))
        garment_image = Image.open(os.path.join(args.garment_dir, garment_file))
        

        model_caption = model_caption_inpaint.get(model_file, "a woman wearing fashion garment.") if args.paired else model_caption_data.get(model_file, "a woman wearing fashion garment.")
        garment_caption = garment_caption_data.get(garment_file, "a fashion garment.")
        prompt = f"The set of three images display model, garment and the model wearing the garment. <IMAGE1> {model_caption} <IMAGE2> {garment_caption} <IMAGE3> <IMAGE1> model wear <IMAGE2> garment."
        
        output = generate_image(pipe, prompt, model_image, garment_image, height=height, width=width, train_double_block_only=args.train_double_block_only)

        if args.repaint:
            mask_file = os.path.join(args.mask_dir, name+"_mask.png")
            source_model_file = os.path.join(args.source_dir, name+".jpg")
            
            # Check if corresponding mask and source files exist
            if not os.path.exists(mask_file) or not os.path.exists(source_model_file):
                print(f"Skipping {model_file} - missing corresponding mask or source image files")
                continue

            mask = Image.open(mask_file).convert('L').resize(output.size)
            source_model = Image.open(source_model_file).resize(output.size)
            output = Image.composite(output, source_model, mask)
        
        output.save(output_path)
        print(f"Generated result for {name}", prompt)
    
    cleanup()

def main():
    args = parse_args()
    # check args
    if args.repaint:
        if args.source_dir is None:
            args.source_dir = args.model_dir
        assert args.mask_dir is not None
    world_size = torch.cuda.device_count()
    mp.spawn(inference, args=(world_size, args), nprocs=world_size)

if __name__ == "__main__":
    main()