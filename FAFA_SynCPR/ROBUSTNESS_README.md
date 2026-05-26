# FAFA Robustness Evaluation

This mode keeps the original clean FAFA inference pipeline and adds robustness rows for corrupted composed queries. The gallery stays clean; perturbations are applied to the reference image and/or text side of each query.

## Run Clean Inference

```bash
cd FAFA_SynCPR
/home/anaconda3/envs/py3.12/bin/python inference_fafa.py \
  --exp-dir output/cpr/FAFA_experiment \
  --model-name tuned_recall_at1_step.pt \
  --itcpr-root /path/to/ITCPR \
  --batch-size 64 \
  --num-workers 2 \
  --device cuda
```

Clean results are still saved as:

```text
<exp-dir>/inference_results_<model-name>.json
```

## Run Robustness Evaluation

```bash
cd FAFA_SynCPR
/home/anaconda3/envs/py3.12/bin/python inference_fafa.py \
  --exp-dir output/cpr/FAFA_experiment \
  --model-name tuned_recall_at1_step.pt \
  --itcpr-root /path/to/ITCPR \
  --batch-size 64 \
  --num-workers 2 \
  --device cuda \
  --robustness-eval \
  --robustness-groups image text conflict \
  --robustness-severities 1 2 3 4 5
```

Outputs:

```text
<exp-dir>/robustness_results_<model-name-without-pt>.csv
<exp-dir>/robustness_results_<model-name-without-pt>.json
```

Each row contains:

```text
corruption_type, severity, R@1, R@5, R@10, R@50
```

## Perturbations

Image corruption: `blur`, `gaussian_noise`, `jpeg_compression`, `brightness_contrast`, `random_occlusion`.

Text perturbation: `synonym`, `paraphrase`, `typo`, `word_deletion`, `color_swap`, `object_swap`.

Modality conflict: `wrong_text` (correct image + wrong text), `wrong_image` (wrong image + correct text), `noisy_text` (correct image + noisy text).

## Quick Debug Run

Use a small query subset to smoke-test the code path:

```bash
/home/anaconda3/envs/py3.12/bin/python inference_fafa.py \
  --exp-dir output/cpr/FAFA_experiment \
  --model-name tuned_recall_at1_step.pt \
  --itcpr-root /path/to/ITCPR \
  --device cuda \
  --robustness-eval \
  --robustness-groups image \
  --robustness-severities 1 \
  --robustness-max-queries 8
```
