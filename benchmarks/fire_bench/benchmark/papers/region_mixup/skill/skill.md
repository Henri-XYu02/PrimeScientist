**Objective**
- Investigate how mixing corresponding spatial regions across training images (region-based mixup) affects generalization and robustness versus whole-image mixup using PreAct ResNet-18 on CIFAR-10.

**Datasets**
- CIFAR-10 (small subset as per instruction) — source: `torchvision.datasets.CIFAR10`.
- Subset size: not specified in the instruction (use a smaller-than-full subset; document actual size when executing).

**Model & Libraries**
- Model: PreAct ResNet-18 (PyTorch implementation).
- Libraries: `torch`, `torchvision`, `numpy`, `matplotlib`, `torchattacks`.

**Augmentations & Methods**
- Base augmentations: standard random crop and horizontal flip.
- Methods to compare (average each over 3 runs):
  - Vanilla Mixup.
  - CutMix (standard single-box variant).
  - Region-based mixup with k in {2, 3, 4} (note: k=1 equals vanilla mixup and is included in the Vanilla baseline): partition into k×k tiles. For each mixed sample, fix a base image A; for each tile j, choose a partner B_j from the batch (B_j ≠ A) and sample a per-tile λ_j ~ Beta(α, α). Optionally use λ_j = max(λ_j, 1-λ_j) to preserve base structure. Mix tiles as x~_j = λ_j·A_j + (1-λ_j)·B_j, and aggregate labels with area-weighted soft targets.
- Loss variants (for each augmentation):
  - Mixup loss only.
  - Mixup loss combined with an additional standard cross-entropy loss term (use CE weight 0.1 with 5-epoch linear warmup).

**Evaluation Metrics**
- Primary: Top-1 test accuracy (report mean over 3 runs).
- Robustness: Post-attack test accuracy under
  - White-box FGSM (from `torchattacks`).
  - Black-box l∞ Square Attack with 100-query budget (from `torchattacks`).

**Reproducible Procedure**
1) Environment setup
   - Install packages: `pip install torch torchvision numpy matplotlib torchattacks`.
   - Set deterministic seeds per run (3 distinct seeds); record them.
2) Data
   - Download CIFAR-10 via `torchvision.datasets.CIFAR10` (train/test splits).
   - Create the “small subset” of the training split (size not specified in instruction; choose and record the exact count at execution time). Always evaluate on the standard CIFAR-10 test split.
   - Apply standard random crop and horizontal flip during training; no augmentation for test.
3) Implement methods
   - Vanilla Mixup: sample per-sample λ_i ~ Beta(α, α) and blend whole images x~_i = λ_i·x_i + (1-λ_i)·x_{perm(i)} with soft labels accordingly.
   - Region-based mixup (k×k):
     - For k in {2,3,4}, partition each image into k×k equal tiles.
     - For each sample i, designate x_i as base A. For each tile j, sample a partner index perm_j(i) ≠ i and a per-tile λ_{i,j} ~ Beta(α, α) (optionally set λ_{i,j} = max(λ_{i,j}, 1-λ_{i,j})).
     - Mix tiles x~_{i,j} = λ_{i,j}·A_{i,j} + (1-λ_{i,j})·B_{perm_j(i),j}.
     - Construct x~_i by assembling x~_{i,j}. Build y~_i by area-weighted sum over tiles with corresponding λ_{i,j}.
4) Loss setups per method
   - Mixup-only loss.
   - Mixup loss + additional cross-entropy term (standard CE on model logits vs ground-truth labels), trained jointly.
5) Training
   - Use full CIFAR-10 training split (no subsampling) to avoid underpowered comparisons.
   - Train PreAct ResNet-18 for 100 epochs with SGD (lr=0.1, momentum=0.9, weight_decay=5e-4), cosine decay, batch_size=128.
   - Evaluate k ∈ {2,3,4}; prioritize k=2–3 as the moderate grid regime shown to work best.
   - For each configuration, run 3 times (different seeds); save best or final checkpoint per run consistently.
6) Accuracy evaluation
   - Compute top-1 test accuracy for each run; report mean across 3 runs per configuration.
7) Robustness evaluation (using models from step 5)
   - FGSM: evaluate post-attack test accuracy using `torchattacks` FGSM.
   - Square Attack (l∞): evaluate post-attack test accuracy using `torchattacks` Square Attack with a 100-query budget.
   - Note: epsilon/attack hyperparameters are not specified in the instruction; record exact values used during execution.
8) Reporting
   - Summarize mean top-1 accuracy (± std if desired) for: Vanilla Mixup, CutMix, Region-mixup k ∈ {2,3,4}; with and without the additional CE term.
   - Summarize robustness results (post-attack accuracies) for FGSM and Square Attack (100 queries) for the same models.

**Ambiguities to Record During Execution (not specified in instruction)**
- Scope of λ sampling (batch vs per-sample vs per-tile). Here, use per-sample (vanilla) and per-sample per-tile (region) λ to improve stability and diversity.

- Subset size for CIFAR-10 training data.
- Optimizer, learning rate schedule, batch size, epochs, weight decay, label smoothing, and any other training hyperparameters.
- FGSM/Square Attack hyperparameters (e.g., ε, step size, iterations for Square Attack beyond the 100-query budget stated).

**Notes**
- The instruction mentions “evaluate ... on all three datasets,” but only CIFAR-10 is specified. Proceed with CIFAR-10 only, as per provided dataset list.

**Execution Notes**
- Run training/eval as modules: use `python -m src.train` and `python -m scripts.run_experiments` instead of `python src/train.py` to avoid `ModuleNotFoundError: No module named 'src'`.
- Ensure `src/__init__.py` exists (e.g., `touch src/__init__.py`), or set `PYTHONPATH=.` when invoking Python.
- Keep the working directory at the project root so the `src` package and `scripts` are importable.
