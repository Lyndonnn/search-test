# Grounded RN03_10 Experiment Report

## Data

- Source rows: 781 clean RN images.
- Hard-pass rows: 620.
- Split: train/dev/test = 458/98/64.
- Review-needed rows: 161, excluded from all SFT training files.
- Package validation: passed.

## GroundingDINO

- Implementation: local official GroundingDINO editable install, not HuggingFace GroundingDINO.
- Checkpoint: `third_party/GroundingDINO_weights/groundingdino_swint_ogc.pth`.
- Best threshold: box=0.1, text=0.1.
- Dev feasibility: mean IoU 0.3042, IoU>=0.3 46.94%, center-hit 70.41%, no-detection 0.00%.
- Test feasibility: mean IoU 0.3766, IoU>=0.3 54.69%, center-hit 73.44%.

## SFT Training

- Base model: `/data/zhengxiang/code/dagig/models/Qwen2.5-VL-7B-Instruct`.
- LoRA: r=32, alpha=64, bf16, gradient checkpointing, effective batch size 16.
- Training: 4 epochs, 116 optimizer steps, lr=2e-5.
- Observed final train loss: 1.639.
- Checkpoint: `checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora`.

## Model Evaluation

- Dev format valid: 98.98%; test format valid: 96.88%.
- Dev model-expression+DINO: mean IoU 0.2828, IoU>=0.3 42.86%, center-hit 64.29%.
- Test model-expression+DINO: mean IoU 0.3102, IoU>=0.3 45.31%, center-hit 59.38%.
- Dev retrieval: R@1 26.53%, R@5 43.88%, MRR 0.3471.
- Test retrieval: R@1 28.12%, R@5 37.50%, MRR 0.3511.
- Dev answer: EM 5.10%, F1 0.1617.
- Test answer: EM 1.56%, F1 0.1573.

## Comparison

- Old direct-bbox autonomous mean IoU: 0.0358; IoU>=0.3: 2.65%.
- New model-expression+DINO dev mean IoU: 0.2828; IoU>=0.3: 42.86%.
- New model-expression+DINO test mean IoU: 0.3102; IoU>=0.3: 45.31%.
- Conclusion: the grounded-expression+DINO route substantially exceeds the old direct numeric bbox route on localization.

## Failure Analysis

- Model ground expressions remain below teacher expressions: token F1 is about 0.58 dev and 0.57 test.
- Model+DINO localization trails teacher+DINO, especially on test center-hit, so expression quality still matters.
- Test has a small final-answer leakage rate of 1.56%; these cases should be inspected before using answer leakage-sensitive rewards.
- Answer EM remains low because this run trained the trajectory format and grounding/search behavior, not a strong end-to-end answerer.

## RL Readiness

Grounding is strong enough to proceed to DAG-IG RL experiments, but RL should reward ground-expression quality, DINO center-hit/IoU, query grounding, and evidence support separately. Do not treat answer EM alone as the main signal yet.

## Next Commands

```bash
cd /storage/zhengxiang/search-test
bash scripts/dagig_train/run_grounded_pipeline_from_zip.sh
```
