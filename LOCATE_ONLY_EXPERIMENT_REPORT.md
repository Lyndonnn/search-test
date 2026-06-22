# Locate-Only Experiment Report

Date: 2026-06-19

This follow-up isolates the full-image `<locate>` failure observed in the autonomous DAG-IG pilot. All code and outputs were produced in `/storage/zhengxiang/search-test`.

## Setup

New scripts:

- `scripts/dagig_train/12_build_locate_only_data.py`
- `scripts/dagig_train/13_eval_locate_only.py`

Locate-only data:

- `data/dagig_locate_only/locate_only_qwen_0_1000_{train,dev,test}.jsonl`
- `data/dagig_locate_only/locate_only_original_pixel_{train,dev,test}.jsonl`
- `data/dagig_locate_only/locate_only_input_pixel_{train,dev,test}.jsonl`

The main supervised setting uses Qwen-style normalized 0-1000 coordinates:

```text
<locate>
[x1, y1, x2, y2]
</locate>
```

Data size:

- Train: 660
- Dev: 132
- Test: 88

## Image Scale Diagnostics

The dataset is dominated by small high-resolution visual targets. Under the 262k pixel processor budget:

| Metric | Value |
| --- | ---: |
| Mean target width at processor scale | 37.33 px |
| Mean target height at processor scale | 26.35 px |
| Min target side < 8 px | 12.7% |
| Min target side < 16 px | 47.6% |
| Min target side < 24 px | 69.5% |
| Min target side < 32 px | 81.8% |

This makes the task much closer to small-text/small-logo grounding than ordinary object detection.

## Base Model Zero-Shot

Model: `/data/zhengxiang/code/dagig/models/Qwen2.5-VL-7B-Instruct`

| Setting | Valid tag | Valid bbox | Mean IoU | Center hit | IoU@0.1 | IoU@0.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base, 262k pixels | 0.0227 | 0.4697 | 0.0011 | 0.0076 | 0.0076 | 0.0000 |
| Base, 1M pixels | 0.0076 | 0.3864 | 0.0022 | 0.0076 | 0.0076 | 0.0000 |

The base model usually answers or explains instead of returning `<locate>`, even with a strong localization-only prompt. Increasing to 1M pixels does not fix zero-shot grounding.

## Locate-Only LoRA

Backbone: Qwen2.5-VL-7B-Instruct  
LoRA rank: 16  
Coordinate target: Qwen 0-1000 normalized bbox

Training runs:

- `checkpoints/dagig_train/locate_only_qwen_0_1000_262k_7b_lora`
  - 262k pixels, 2 epochs, final train loss 0.9522
- `checkpoints/dagig_train/locate_only_qwen_0_1000_1mp_7b_lora`
  - 1M pixels, 1 epoch, final train loss 1.007

Dev results:

| Setting | Valid tag | Valid bbox | Mean IoU | Center hit | IoU@0.1 | IoU@0.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA 262k train / 262k infer | 1.0000 | 1.0000 | 0.0019 | 0.0152 | 0.0076 | 0.0000 |
| LoRA 1M train / 1M infer | 1.0000 | 1.0000 | 0.0013 | 0.0152 | 0.0076 | 0.0000 |
| LoRA 262k train / 1M infer | 1.0000 | 0.9924 | 0.0042 | 0.0076 | 0.0227 | 0.0000 |

Train-subset sanity:

| Setting | Split | n | Mean IoU | IoU@0.3 |
| --- | --- | ---: | ---: | ---: |
| LoRA 262k train / 262k infer | train subset | 132 | 0.0031 | 0.0076 |

The LoRA learns the output format perfectly but does not learn reliable visual grounding. The train-subset result is also near zero, so this is not just dev-set generalization failure.

## Prediction Pattern

Gold dev boxes in Qwen 0-1000 coordinates:

- Gold center mean: `(532.6, 567.8)`
- Gold center stdev: `(248.3, 200.8)`
- Gold box size mean: `62.8 x 56.9`

LoRA 262k predictions:

- Predicted center mean: `(516.7, 602.9)`
- Predicted center stdev: `(201.8, 145.9)`
- Predicted box size mean: `41.5 x 25.8`

The model outputs plausible-looking boxes and a broad position prior, but it rarely overlaps the target. It is not merely a formatting issue or a coordinate parser issue.

## Interpretation

The locate failure is multi-factor:

1. The base model does not naturally follow this locate-only bbox task on Pix2Fact images.
2. Qwen 0-1000 coordinates fix the coordinate-convention mismatch, but they are not sufficient.
3. 262k pixel compression makes most targets extremely small.
4. 1M pixels alone still does not solve the problem, which suggests the prompt/data distribution and grounding supervision are also limiting.
5. The current 660-example locate SFT is not enough for robust small-sign/small-logo localization.

The strongest evidence is the train-subset sanity result: even on seen training examples, the 262k locate-only adapter has only 0.0076 IoU@0.3.

## Recommended Next Step

Do not spend more GPU on end-to-end autonomous RL yet. The next useful experiment is a two-stage high-resolution locator:

1. Build tile-based locate data from full images.
2. Ask the model to choose a coarse tile first, then predict a local 0-1000 bbox inside that tile.
3. Crop the predicted region and feed it into the already working oracle-crop DAG-IG pipeline.

This directly attacks the small-target compression problem and avoids asking one low-resolution full-image pass to localize tiny visual anchors.
