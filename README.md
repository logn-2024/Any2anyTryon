# Any2AnyTryon: Leveraging Adaptive Position Embeddings for Versatile Virtual Clothing Tasks
<div style="display: flex; justify-content: center; align-items: center;">
  <a href="https://arxiv.org/abs/2501.15891" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-2501.15891-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href='https://logn-2024.github.io/Any2anyTryon/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a>
  <a href="https://huggingface.co/spaces/jamesliu1217/Any2anyTryon_exp" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a>
  <a href='https://huggingface.co/loooooong/Any2anyTryon' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Hugging Face-ckpts-orange?style=flat&logo=HuggingFace&logoColor=orange' alt='huggingface'>
  </a>
  <a href='https://huggingface.co/datasets/loooooong/LAION-Garment' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Hugging Face-datasets-orange?style=flat&logo=HuggingFace&logoColor=orange' alt='huggingface'>
  </a>
</div>

![teaser](asset/images/teaser.png)

## Environment
```bash
conda create -n Any2AnyTryon python=3.11
conda activate Any2AnyTryon
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade -r requirements.txt
```

## Demo
Using `--group_offloading` saves memory but slows processing.
```bash
python app.py
```

## Inference
Choose a LoRA from [here](https://huggingface.co/loooooong/Any2anyTryon) or use `--lora_name` with the script.
```bash
python infer.py --model_image ./asset/images/model/model1.png --garment_image ./asset/images/garment/garment1.jpg \
--prompt="<MODEL> a man <GARMENT> t-shirt with pockets <TARGET> man wearing the t-shirt"
```

## Test

Download the dataset from the [VITON-HD](https://github.com/shadow2496/VITON-HD). Generate segmentation masks using [AutoMasker](https://github.com/Zheng-Chong/CatVTON/blob/edited/preprocess_agnostic_mask.py) for repainting. For unpaired data evaluation, execute the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python test_vitonhd.py --model_path black-forest-labs/FLUX.1-dev \
--model_dir data/zalando-hd-resized/test/image --garment_dir data/zalando-hd-resized/test/cloth \
--output_dir ./results/vitonhd_test_unpaired_repaint --meta_file data/zalando-hd-resized/test_pairs.txt \
--mask_dir data/zalando-hd-resized/test/mask --source_dir data/zalando-hd-resized/test/image \
--train_double_block_only --repaint 
```
For paired test, download [test set](https://huggingface.co/datasets/loooooong/Any2anyTryon_vitonhd_test) and extract images to local directory.
```bash
python src/download_data.py
```
```bash
CUDA_VISIBLE_DEVICES=0 python test_vitonhd.py --model_path black-forest-labs/FLUX.1-dev \
--model_dir data/zalando-hd-resized/test/image_synthesis --garment_dir data/zalando-hd-resized/test/cloth \
--output_dir ./results/vitonhd_test_paired_repaint --meta_file data/zalando-hd-resized/test_pairs.txt \
--mask_dir data/zalando-hd-resized/test/mask --source_dir data/zalando-hd-resized/test/image \
--train_double_block_only --repaint --paired
```

## Datasets
Some data can be downloaded from [here](https://huggingface.co/datasets/loooooong/LAION-Garment). Use the following code to ensure pixel correspondence between the original image and the synthesized image:
```python
from src.utils import crop_to_multiple_of_16
img_src = Image.open(<img_path>)
img_inpaint = Image.open(io.BytesIO(<inpaint_bytes>))

img_src = crop_to_multiple_of_16(img_src)
assert img_src.size==img_inpaint.size
```

## To-Do List
- \[x\] Demo code and gradio interface
- \[x\] Inference code
- \[x\] Tryon checkpoint
- \[x\] Model generation checkpoint
- \[x\] Garment reconstruction checkpoint
- \[x\] Base all tasks checkpoint
- \[x\] Dataset preparation
- \[x\] Training code

## Citation

```bibtex
@misc{guo2025any2anytryonleveragingadaptiveposition,
    title={Any2AnyTryon: Leveraging Adaptive Position Embeddings for Versatile Virtual Clothing Tasks}, 
    author={Hailong Guo and Bohan Zeng and Yiren Song and Wentao Zhang and Chuang Zhang and Jiaming Liu},
    year={2025},
    eprint={2501.15891},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2501.15891}, 
}
```