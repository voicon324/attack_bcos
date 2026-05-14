# CamoPatch

Code for the paper "CamoPatch: An Evolutionary Strategy for Generating Camouflaged Adversarial Patches" published at NeurIPS 2023.

## Faster local runs

The default run keeps the original sequential `(1 + 1)` evolutionary strategy and CPU execution used by the original code:

```bash
python ConCamoPatch.py \
  --model 1 \
  --model_source bcos \
  --image_dir 8.JPEG \
  --true_label <imagenet_label> \
  --save_directory results/resnet50_8 \
  --queries 10000
```

Add an L-infinity bound with `--linf`. The images are normalized to `[0,1]`, so `8/255` is accepted directly:

```bash
python ConCamoPatch.py \
  --model 1 \
  --model_source bcos \
  --image_dir 8.JPEG \
  --true_label <imagenet_label> \
  --save_directory results/resnet50_8_linf \
  --queries 10000 \
  --linf 8/255
```

For faster wall-clock time on a GPU, evaluate multiple patch candidates in one model forward:

```bash
python ConCamoPatch.py \
  --model 1 \
  --model_source bcos \
  --image_dir 8.JPEG \
  --true_label <imagenet_label> \
  --save_directory results/resnet50_8_fast \
  --device cuda \
  --batch_size 8 \
  --trace_every 0
```

To choose the initial patch position with the B-cos contribution-margin rule from
`attacks/explain_guided_circle_rgb_es_patch.py` and keep that position fixed:

```bash
python ConCamoPatch.py \
  --model 1 \
  --model_source bcos \
  --image_dir 8.JPEG \
  --true_label <imagenet_label> \
  --save_directory results/resnet50_8_fixed_bcos_pos \
  --fixed_bcos_position
```

To use that B-cos position only as the initial location, while still allowing
CamoPatch's normal location updates:

```bash
python ConCamoPatch.py \
  --model 1 \
  --model_source bcos \
  --image_dir 8.JPEG \
  --true_label <imagenet_label> \
  --save_directory results/resnet50_8_init_bcos_pos \
  --init_bcos_position
```

Notes:

- `--model_source auto` uses local/uploaded B-cos weights for `--model 1` (ResNet-50). Use `--model_source torchvision` to run the original torchvision CamoPatch model loading path.
- Runs save `attack_model`, `attack_model_source`, and `attack_model_index` in the `.npy` result; `--model 1 --model_source bcos` records `attack_model=bcos_resnet50`.
- Torchvision/original weights are loaded offline from `weights/torchvision-imagenet` or an attached Kaggle dataset containing `torchvision-imagenet/`.
- CamoPatch uses the same `Resize(224) -> CenterCrop(224)` spatial preprocessing as `evaluate_original_models_on_csv.py`; torchvision normalization is applied inside `ImageNetModel.predict`.
- `--batch_size 1` preserves the original sequential query order.
- `--batch_size > 1` keeps the same CamoPatch objective and acceptance logic, but evaluates parallel candidates from the current state, so it is a faster batched variant rather than a bit-for-bit identical run.
- `--device cuda` uses the same sequential algorithm when `--batch_size 1`, but GPU floating-point math can differ slightly from CPU.
- `--s` controls patch width and height in pixels after the 224x224 crop. The default is `16`, so the default patch is `16x16`.
- `--linf` projects every rendered patch candidate into the per-pixel box `x_orig +/- linf` before evaluation, and re-projects when the patch location changes.
- `--fixed_bcos_position` disables CamoPatch location updates and uses one B-cos contribution-margin patch position for the full run.
- `--init_bcos_position` starts from the B-cos contribution-margin patch position but keeps CamoPatch location updates enabled.
- `--trace_every 0` disables saving every genotype/location step, which reduces Python overhead and `.npy` output size.
- Each run saves `<save_directory>.npy`, `<save_directory>_adversary.png`, and `<save_directory>_patch.png` by default. Add `--no_save_images` to skip PNG exports.

Run the ResNet-50 L-infinity sweep over the used-image CSV:

```bash
./run_resnet50_linf_sweep.sh ../used_images_500_1.csv results/camopatch_resnet50 \
  --device cuda \
  --batch_size 8 \
  --trace_every 0
```

The sweep script defaults to original torchvision ResNet-50. Set `MODEL_SOURCE=bcos` to run the same CSV sweep on B-cos ResNet-50.

Run multiple CSV images in one model forward while keeping strict per-image
`(1+1)-ES` semantics:

```bash
python CamoPatch/ConCamoPatchBatch.py \
  --images_csv data/used_images_500_local.csv \
  --save_root artifacts/outputs/camopatch_bcos_strict_1p1_full_linf32_256 \
  --model 1 \
  --model_source bcos \
  --device cuda \
  --image_batch_size 8 \
  --queries 10000 \
  --init_bcos_position \
  --linf 32/256
```

Here `--image_batch_size` is the number of images evaluated together. The
script still creates exactly one candidate per image per query, so each image
uses strict `(1+1)-ES`; batching is only across different images.
