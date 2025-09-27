import torch
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
import random
import cv2
import os
import io
import numpy as np

import pyarrow.parquet as pq
from tqdm import tqdm

from src.utils import resize_and_pad_to_size, crop_to_multiple_of_16

def safe_load_npz(path):
    """Safely load npz file, return None if file doesn't exist"""
    if path is not None and os.path.exists(path):
        try:
            return np.load(path)
        except:
            print(f"Error loading npz file: {path}")
            return None
    return None

def safe_read_image(path):
    try:
        return Image.open(path).convert("RGB")
    except:
        with open("error.txt", "a") as f:
            f.write(f"Image processing error: {path}\n")
        return Image.fromarray(np.zeros((512, 384, 3), dtype=np.uint8))
    

Image.MAX_IMAGE_PIXELS = None

def make_train_dataset(args, tokenizer, accelerator=None):
    if args.train_data_path is not None:
        print("load_data")
        dataset = load_dataset(
            'json', 
            data_files=args.train_data_path
        )

    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.caption_column is None:
        caption_column = column_names[0]
        print(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    tokenizer_clip = tokenizer[0]
    tokenizer_t5 = tokenizer[1]

    p = 0. #0.1
    def tokenize_prompt_clip_t5(examples):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                if random.random() < p:
                    captions.append("")
                else:
                    captions.append(caption)
            elif isinstance(caption, list):
                if random.random() < p:
                    captions.append(" ")
                else:
                    captions.append(random.choice(caption))
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        # print(captions)
        text_inputs = tokenizer_clip(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_1 = text_inputs.input_ids

        text_inputs = tokenizer_t5(
            captions,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs.input_ids
        return text_input_ids_1, text_input_ids_2
    
    MIN_SIZE, MAX_SIZE = 384, 896
    def resize_and_adjust(image, max_size=MAX_SIZE, multiple_of=16):
        width, height = image.size
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            new_width, new_height = int(width * scale), int(height * scale)
        else:
            new_width, new_height = width, height
        new_width = (new_width // multiple_of) * multiple_of
        new_height = (new_height // multiple_of) * multiple_of
        return transforms.Resize((new_height, new_width), interpolation=transforms.InterpolationMode.BILINEAR)(image)

    train_transforms = transforms.Compose(
        [
            # transforms.Lambda(lambda img: resize_and_adjust(img, max_size=args.noise_size, multiple_of=16)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    size = args.cond_size
    cond_train_transforms = transforms.Compose(
        [
            # transforms.Lambda(lambda img: resize_and_adjust(img, max_size=args.cond_size, multiple_of=16)),
            # transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def random_resize(image, width_min=MIN_SIZE, height_min=MIN_SIZE, width_max=MAX_SIZE, height_max=MAX_SIZE, is_tryon=True):
        width, height = image.width, image.height
        assert width % 16 == 0 and height % 16 == 0, "width and height must be multiples of 16"
        # assert width_min <= width <= width_max, f"width ({width}) must be in [{width_min}, {width_max}]"
        # assert height_min <= height <= height_max, f"height ({height}) must be in [{height_min}, {height_max}]"

        # if width==height or ( not (width==768 and height==1024)):
        if is_tryon:
            if width>768 and height>768 and width/height>0.75 and width/height<4./3: #width*height>602112:#672*896:
                width_max, height_max = 768, 768
        else:
            width_max, height_max = 1024, 1024
        need_pad = False
        tgt_width, tgt_height = width, height
        if width > width_max:
            need_pad, tgt_width = True, width_max
        elif width < width_min:
            need_pad, tgt_width = True, width_min
        if height > height_max:
            need_pad, tgt_width = True, height_max
        elif height < height_min:
            need_pad, tgt_width = True, height_min
        image = resize_and_pad_to_size(image, tgt_width, tgt_height)[0] if need_pad else image
        width, height = image.width, image.height
        
        possible_resolutions = []
        
        for w in range(width_min, min(width, width_max) + 1, 16):
            if w * height % width != 0:
                continue
            h = w * height // width
            if h % 16 != 0:
                continue
            if h < height_min or h > min(height, height_max):
                continue
            if w * height == h * width:
                possible_resolutions.append((w, h))
        if len(possible_resolutions)==0:
            possible_resolutions.append((width, height))

        return image.resize(random.choice(possible_resolutions))

    def preprocess_train(examples):
        _examples = {}        

        # # fix resolution
        # target_images = [Image.open(img).convert("RGB").resize((384,512)) for img in examples["target_image"]]  
        # condspa_images = [Image.open(img).convert("RGB").resize((384,512)) if img is not None else None for img in examples["condition_spatial_image"]]
        # condsub_images = [Image.open(img).convert("RGB").resize((384,512)) if img is not None else None for img in examples["condition_subject_image"]] 

        # random resolution
        all_tryon_marks = [(x is not None) and (y is not None) and (z is not None) for x,y,z in zip(examples["target_image"], examples["condition_spatial_image"],examples["condition_subject_image"])]
        target_images = [
            random_resize(
                crop_to_multiple_of_16(resize_and_adjust(
                    crop_to_multiple_of_16(
                        safe_read_image(img)
                    )
                )),
                is_tryon=is_tryon
            )
            for img, is_tryon in zip(examples["target_image"], all_tryon_marks)
        ]
        condspa_images = [random_resize(crop_to_multiple_of_16(resize_and_adjust(safe_read_image(img))), is_tryon=is_tryon) if img is not None else None for img,is_tryon in zip(examples["condition_spatial_image"], all_tryon_marks)]
        condsub_images = [random_resize(crop_to_multiple_of_16(resize_and_adjust(safe_read_image(img))), is_tryon=is_tryon) if img is not None else None for img,is_tryon in zip(examples["condition_subject_image"], all_tryon_marks)]

        examples[caption_column] = [p or "" for p in examples[caption_column]]
        _examples["condspa_pixel_values"] = [(cond_train_transforms(image) if image else None) for image in condspa_images]
        _examples["condsub_pixel_values"] = [(cond_train_transforms(image) if image else None) for image in condsub_images]
        _examples["pixel_values"] = [train_transforms(image) for image in target_images]
        _examples["token_ids_clip"], _examples["token_ids_t5"] = tokenize_prompt_clip_t5(examples)
        return _examples

    if accelerator is not None:
        with accelerator.main_process_first():
            train_dataset = dataset["train"].with_transform(preprocess_train)
    else:
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    condspa_pixel_values = torch.stack([example["condspa_pixel_values"] for example in examples]).to(memory_format=torch.contiguous_format).float() if examples[0]["condspa_pixel_values"] is not None else None
    condsub_pixel_values = torch.stack([example["condsub_pixel_values"] for example in examples]).to(memory_format=torch.contiguous_format).float() if examples[0]["condsub_pixel_values"] is not None else None
    target_pixel_values = torch.stack([example["pixel_values"] for example in examples])
    target_pixel_values = target_pixel_values.to(memory_format=torch.contiguous_format).float()
    token_ids_clip = torch.stack([example["token_ids_clip"] for example in examples])
    token_ids_t5 = torch.stack([example["token_ids_t5"] for example in examples])

    return {
        "condspa_pixel_values": condspa_pixel_values,
        "condsub_pixel_values": condsub_pixel_values,
        "pixel_values": target_pixel_values,
        "text_ids_1": token_ids_clip,
        "text_ids_2": token_ids_t5,
    }
