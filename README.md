# Multi-Species Weed Recognition: 9-Class DeepWeeds Image Classification

A reproducible PyTorch study of multiclass weed/background recognition on the **DeepWeeds** dataset. The implementation compares three ImageNet-pretrained vision architectures—**EfficientNet-B0**, **MobileNetV3-Large**, and **ViT-Tiny**—under a common 224 × 224 training and evaluation protocol, with class-balanced cross-entropy, staged transfer learning, validation-driven configuration selection, leakage auditing, held-out test evaluation, error analysis, and automatic artifact packaging.

> **Research status.** The numerical results reported in this README are the observed outputs of the provided notebook execution accompanying this source file. They are not newly generated during documentation. The project intentionally avoids fabricated benchmark values and does not claim exact reproduction of the DeepWeeds repository's published fold assignments.

## Abstract

We study nine-class image classification for invasive-weed recognition, where the target label is one of eight weed species or a **Negative** background class. The implementation uses the official DeepWeeds `labels.csv` annotation file, resolves filenames to image files, derives conservative acquisition-based grouping keys from the documented DeepWeeds filename convention, and constructs one deterministic **60/20/20 train/validation/test geometry** from five stratified group folds.

Three pretrained architectures are evaluated under matched image resolution, training augmentation, class-balanced loss, staged freezing/fine-tuning, and validation-only model selection. The default run uses a controlled two-configuration pilot for each architecture, selects the configuration with the best validation top-1 accuracy, trains the selected configuration on the full available training partition, restores the best validation checkpoint, and only then evaluates the held-out test set.

The observed run achieved a best held-out top-1 accuracy of **84.02%** with **MobileNetV3-Large**. ViT-Tiny obtained the highest macro F1 (**80.32%**) among the three models, while EfficientNet-B0 obtained the highest macro recall (**88.01%**). These results should be interpreted as a single-seed experiment on one leakage-audited 60/20/20 partition, not as a five-fold statistical estimate.

## Research Questions

The experiment is designed to answer three practical questions:

1. How do a compact convolutional model, a larger lightweight CNN, and a small Vision Transformer compare under the same DeepWeeds protocol?
2. Does a conservative grouped split materially protect the evaluation from acquisition-related leakage compared with an image-level random split?
3. How does model behavior differ when overall accuracy is considered together with macro-averaged precision, recall, F1, confusion structure, and high-confidence errors?

## Dataset

The project uses the public **DeepWeeds** dataset, which contains **17,509 images** spanning eight invasive weed species plus a negative/background class. The official repository documents filenames in the form `YYYYMMDD-HHMMSS-ID`, where the final integer identifies the imaging instrument, and it provides the label subsets used by the original five-fold evaluation protocol.

### Class distribution

| Internal code | Class          | Images |  Share |
| ------------- | -------------- | -----: | -----: |
| `DW_NEG`      | Negative       |  9,106 | 52.01% |
| `DW_CHAP`     | Chinee apple   |  1,125 |  6.43% |
| `DW_SIWE`     | Siam weed      |  1,074 |  6.13% |
| `DW_LANT`     | Lantana        |  1,064 |  6.08% |
| `DW_PRAC`     | Prickly acacia |  1,062 |  6.07% |
| `DW_PARK`     | Parkinsonia    |  1,031 |  5.89% |
| `DW_PART`     | Parthenium     |  1,022 |  5.84% |
| `DW_SNWE`     | Snake weed     |  1,016 |  5.80% |
| `DW_RUBV`     | Rubber vine    |  1,009 |  5.76% |

The strong class imbalance—especially the dominance of the `DW_NEG` class—is handled during training with class-balanced cross-entropy weights computed **from the training partition only**.

### Data integrity and label resolution

The acquisition layer is deliberately defensive. It validates candidate image archives by checking ZIP structure, the expected **17,509 image count**, and ZIP CRC integrity; the Zenodo fallback additionally verifies the expected file size and MD5 checksum. The pipeline also checks that every labeled filename resolves to an image, that the official numeric labels agree with the mapped species names, and that all nine target classes are present.

The normal metadata path does **not** open all images. It therefore reports image dimensions as unavailable and marks the metadata status as `not_scanned`. Actual image decoding occurs later inside the PyTorch dataset during training and evaluation.

## Scientific Protocol

### Leakage-aware splitting

The split is derived from `StratifiedGroupKFold` with `n_splits = 5` and a deterministic seed sweep. The intended geometry is:

* fold 0 → test
* fold 1 → validation
* folds 2–4 → training

This produces a nominal 60/20/20 partition while enforcing non-overlap of the selected groups across partitions.

DeepWeeds does not expose the same plant/box metadata used by some other agricultural datasets. The implementation therefore constructs grouping candidates in descending semantic strength:

1. `box_id`: capture date + instrument ID
2. exact capture timestamp: `YYYYMMDD-HHMMSS`
3. unique image identifier as a strict singleton fallback

A candidate grouping is accepted only when every class has support in all five folds. The recorded run selected **`capture_timestamp`** with seed **42**. The resulting split contains:

| Partition  | Images |  Share |
| ---------- | -----: | -----: |
| Train      | 10,495 | 59.94% |
| Validation |  3,509 | 20.04% |
| Test       |  3,505 | 20.02% |

The run reported **17,204 groups** and zero group overlap between train/validation, train/test, and validation/test.

> **Important methodological limitation.** Because 17,204 groups are formed from 17,509 images, the selected timestamp grouping is close to an image-level split for much of the dataset. It is still a stronger safeguard than ignoring acquisition structure entirely, but it should not be interpreted as proof that all near-duplicate or temporally correlated imagery has been eliminated.

The exact split is saved to `results/configs/split_metadata.csv`, with a SHA-256 checksum stored in `results/configs/split_metadata.sha256`. The split configuration is also serialized to `results/configs/split_config.json`.

### Input preprocessing

All three models consume **224 × 224 RGB** tensors.

Training uses mild stochastic augmentation:

* `RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1))`
* horizontal flip with probability 0.5
* `ColorJitter` with brightness 0.15, contrast 0.15, saturation 0.10, hue 0.02
* `RandomAffine` with ±10° rotation, translation up to 3%, and scale 0.95–1.05
* tensor conversion and model-specific normalization

Evaluation uses resize-then-center-crop geometry with an effective resize dimension of **256 pixels** for a 224-pixel crop, followed by tensor conversion and normalization.

The CNNs use ImageNet normalization:

$$
\mu = (0.485,\;0.456,\;0.406),
\qquad
\sigma = (0.229,\;0.224,\;0.225).
$$

The configured `timm` ViT-Tiny checkpoint uses:

$$
\mu = (0.5,\;0.5,\;0.5),
\qquad
\sigma = (0.5,\;0.5,\;0.5).
$$

## Methodology

### Model comparison

The classification head of each pretrained network is replaced with a nine-logit output layer:

| Model             | Implementation                                               | Pretraining                  | Trainable layers during fine-tuning                            |
| ----------------- | ------------------------------------------------------------ | ---------------------------- | -------------------------------------------------------------- |
| EfficientNet-B0   | `torchvision.models.efficientnet_b0`                         | ImageNet-1K                  | classifier + last 2 feature blocks                             |
| MobileNetV3-Large | `torchvision.models.mobilenet_v3_large`                      | ImageNet-1K                  | classifier + last 3 feature blocks                             |
| ViT-Tiny          | `timm.create_model("vit_tiny_patch16_224", pretrained=True)` | pretrained `timm` checkpoint | head/fc_norm + last 2 transformer blocks + final normalization |

The observed parameter counts after replacing the classification head were:

| Model             | Total parameters |
| ----------------- | ---------------: |
| EfficientNet-B0   |        4,019,077 |
| MobileNetV3-Large |        4,213,561 |
| ViT-Tiny          |        5,526,153 |

The code intentionally uses the architectures' native pretrained normalization profiles rather than applying one normalization rule indiscriminately.

### Two-stage transfer learning

Each experiment begins with pretrained weights and follows two stages.

**Stage 1 — head training.** All backbone parameters are frozen and only the classifier head is optimized.

**Stage 2 — selective fine-tuning.** A small architecture-specific portion of the backbone is unfrozen. The classifier and backbone are optimized with separate learning rates, implementing discriminative fine-tuning.

The optimizer is `AdamW` in both stages. A cosine annealing scheduler is created independently for each stage. During the head-only stage, frozen convolutional BatchNorm layers are switched to evaluation mode so their running statistics are not updated without trainable parameters.

### Loss function

The training criterion is weighted cross-entropy. For a sample with target class $y_i$ and logits $z_i \in \mathbb{R}^{C}$, the implementation first forms the standard softmax probabilities

$$
p_{ic} = \frac{e^{z_{ic}}}{\sum_{j=1}^{C} e^{z_{ij}}}.
$$

For a minibatch of $N_b$ samples, PyTorch's default `CrossEntropyLoss(weight=...)` mean reduction gives

$$
\mathcal{L}_{\mathrm{batch}} =
\frac{
\sum_{i=1}^{N_b}
w_{y_i}\left(-\log p_{i,y_i}\right)
}{
\sum_{i=1}^{N_b} w_{y_i}
}.
$$

The class weight for class $c$ is computed as

$$
\tilde{w}_c = \frac{N}{C\,N_c},
$$

with $N$ the number of training samples, $C=9$ the number of classes, and $N_c$ the training count of class $c$. The vector is then normalized by its own mean so that the average class weight is 1:

$$
w_c =
\frac{\tilde{w}_c}{
\frac{1}{C}
\sum_{k=1}^{C}\tilde{w}_k
}.
$$

This changes the relative contribution of classes while keeping the overall weight scale controlled. Importantly, the code computes the weights from `train_df` only; validation and test labels are not used to construct the loss.

### Hyperparameter selection

For each architecture, the code evaluates two learning-rate configurations on a **6,000-image stratified pilot subset** when the default `colab_balanced` profile is active. The pilot never uses the test set for selection.

| Model             | Pilot configuration | Head LR | Fine-tune LR | Weight decay |
| ----------------- | ------------------- | ------: | -----------: | -----------: |
| EfficientNet-B0   | `lr1e-3`            |    1e-3 |         3e-4 |         1e-4 |
| EfficientNet-B0   | `lr5e-4`            |    5e-4 |         1e-4 |         1e-4 |
| MobileNetV3-Large | `lr1e-3`            |    1e-3 |         3e-4 |         1e-4 |
| MobileNetV3-Large | `lr5e-4`            |    5e-4 |         1e-4 |         1e-4 |
| ViT-Tiny          | `lr3e-4`            |    3e-4 |         1e-4 |         1e-2 |
| ViT-Tiny          | `lr1e-4`            |    1e-4 |         5e-5 |         1e-2 |

The selection rule is deterministic: choose the configuration with the highest validation top-1 accuracy, breaking ties by choosing the smaller best epoch. In the recorded run, the selected configurations were `lr1e-3` for EfficientNet-B0 and MobileNetV3-Large, and `lr3e-4` for ViT-Tiny. The two MobileNetV3-Large pilot configurations tied at approximately **74.75%** validation top-1, so the earlier-best-epoch tie-break selected `lr1e-3` deterministically.

### Final training and checkpoint selection

The selected configuration is retrained on the full available training partition. Under the recorded `colab_balanced` run, `FINAL_MAX_TRAIN_SAMPLES = 120,000`, which exceeds the 10,495 training images actually available, so the entire training partition is used.

The default schedule is:

* 1 head-training epoch
* up to 6 fine-tuning epochs
* early stopping patience of 2 fine-tuning epochs
* minimum validation improvement (`MIN_DELTA`) of `1e-4`

The checkpoint with the best validation top-1 accuracy is restored before the held-out test evaluation.

## Mathematical Details

### Top-1 and top-5 accuracy

For logits $z_i$, let $\operatorname{TopK}(z_i,k)$ denote the indices of the $k$ largest logits. The batch top-$k$ accuracy is the mean indicator

$$
\operatorname{Acc}@k =
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
\left[
y_i \in \operatorname{TopK}(z_i,k)
\right].
$$

The implementation computes top-1 and top-5 counts together from one `torch.topk` operation per batch and aggregates those counts on the device.

### Test-set metrics

The final report includes:

* **Accuracy:** fraction of test examples classified correctly.
* **Macro precision:** arithmetic mean of the nine class-level precision values.
* **Macro recall:** arithmetic mean of the nine class-level recall values.
* **Macro F1:** arithmetic mean of the nine class-level F1 values.
* **Weighted F1:** support-weighted mean of class-level F1 values.

For class $c$, precision, recall, and F1 follow the standard definitions:

$$
\operatorname{Precision}_c =
\frac{TP_c}{TP_c + FP_c},
\qquad
\operatorname{Recall}_c =
\frac{TP_c}{TP_c + FN_c},
$$

$$
F1_c =
\frac{
2\,\operatorname{Precision}_c\,\operatorname{Recall}_c
}{
\operatorname{Precision}_c + \operatorname{Recall}_c
}.
$$

The macro F1 reported by the code is the unweighted average of the nine class-level values, while weighted F1 uses test-set class support as weights.

## Experimental Results

### Final held-out comparison

The following values are taken directly from the recorded execution of the supplied notebook.

| Model                 | Parameters |   Test Top-1 | Macro Precision | Macro Recall |     Macro F1 |  Weighted F1 | Best Val Top-1 | Best Epoch | Training Time (s) |
| --------------------- | ---------: | -----------: | --------------: | -----------: | -----------: | -----------: | -------------: | ---------: | ----------------: |
| **MobileNetV3-Large** |  4,213,561 | **0.840228** |        0.824766 |     0.794806 |     0.797392 | **0.836726** |   **0.846965** |          5 |           1011.26 |
| ViT-Tiny              |  5,526,153 |     0.826534 |        0.767189 |     0.865549 | **0.803243** |     0.831901 |       0.823312 |          7 |           1203.45 |
| EfficientNet-B0       |  4,019,077 |     0.818830 |        0.750602 | **0.880062** |     0.801073 |     0.823379 |       0.820747 |          6 |           1044.23 |

The primary selection criterion for model comparison is held-out **test top-1 accuracy**, as implemented by the final sorting step. Under that criterion, MobileNetV3-Large is the best model in this run.

At the same time, the metrics show why accuracy alone is insufficient: ViT-Tiny achieves the highest macro F1, while EfficientNet-B0 achieves the highest macro recall. The three models therefore exhibit meaningfully different error profiles despite their relatively close overall accuracies.

### Pilot selection results

| Model             | Selected setting | Best pilot validation top-1 | Pilot runtime (s) |
| ----------------- | ---------------- | --------------------------: | ----------------: |
| EfficientNet-B0   | `lr1e-3`         |                    0.670846 |            883.29 |
| MobileNetV3-Large | `lr1e-3`         |                    0.747506 |            178.49 |
| ViT-Tiny          | `lr3e-4`         |                    0.634939 |            207.93 |

The pilot stage is deliberately a **small configuration search**, not an exhaustive hyperparameter optimization study.

### Per-class behavior

The recorded test reports show distinct strengths and weaknesses.

For **EfficientNet-B0**, the lowest-recall classes were `DW_CHAP` (72.57%) and `DW_NEG` (74.95%), while `DW_SIWE` reached 97.25% recall.

For **MobileNetV3-Large**, recall was lowest for `DW_PART` (57.75%), `DW_CHAP` (58.85%), and `DW_PRAC` (64.29%); `DW_SIWE` reached 96.33% recall.

For **ViT-Tiny**, recall was lowest for `DW_PART` (70.89%) and `DW_NEG` (78.17%), while `DW_SIWE` reached 96.79% recall.

These figures demonstrate a recurring asymmetry: some models recognize several weed classes well but trade away recall on visually difficult classes, and the negative/background class is not uniformly easy despite having more than half of the dataset.

### Error analysis

The implementation saves full class-probability vectors and derives high-confidence incorrect predictions from the predicted-class probability. The most frequent observed confusion pairs for the best test model, MobileNetV3-Large, included:

| True class | Predicted class | Count |
| ---------- | --------------- | ----: |
| `DW_PART`  | `DW_NEG`        |    71 |
| `DW_PRAC`  | `DW_NEG`        |    58 |
| `DW_NEG`   | `DW_SIWE`       |    50 |
| `DW_PARK`  | `DW_NEG`        |    40 |
| `DW_NEG`   | `DW_LANT`       |    39 |
| `DW_CHAP`  | `DW_NEG`        |    36 |
| `DW_CHAP`  | `DW_SNWE`       |    36 |
| `DW_NEG`   | `DW_SNWE`       |    31 |
| `DW_NEG`   | `DW_RUBV`       |    31 |

The analysis intentionally stops short of assigning biological causes to these confusions. The current evidence supports an empirical description of model behavior, not a causal explanation of why specific species are confused.

### Runtime

The recorded execution ran on a **Tesla T4** using CUDA with `torch.bfloat16` automatic mixed precision. Logged final-model training plus validation time was approximately **3,252.1 seconds** in total across the three architectures, with individual end-to-end training times of roughly 1,011–1,203 seconds.

The pilot stage was substantially more expensive for EfficientNet-B0 in the recorded environment because its first pilot run was dominated by one slow initial epoch; this is a runtime observation, not a model-quality conclusion.

## Implementation Details

### Software stack

The validated notebook execution reported:

| Component    | Recorded version |
| ------------ | ---------------- |
| Python       | 3.13.15          |
| PyTorch      | 2.11.0+cu128     |
| Torchvision  | 0.26.0+cu128     |
| NumPy        | 2.1.3            |
| pandas       | 2.2.3            |
| scikit-learn | 1.6.1            |
| timm         | 1.0.28           |

The source also installs `timm` on demand when it is missing, within the constraint `timm>=1.0,<2.0`.

### Core Python components

The implementation is organized around the following responsibilities:

| Component                                        | Role                                                                                      |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| `prepare_dataset_root()` and acquisition helpers | dataset discovery, caching, downloading, integrity checks, and extraction                 |
| metadata construction                            | official-label resolution, class mapping, duplicate-path removal, grouping metadata       |
| `make_grouped_split()`                           | deterministic stratified group-fold assignment and leakage checks                         |
| `OPPDPlantDataset`                               | lazy image loading and per-model preprocessing                                            |
| `build_model()`                                  | pretrained EfficientNet-B0, MobileNetV3-Large, or ViT-Tiny construction                   |
| `freeze_for_stage()`                             | architecture-specific freezing and fine-tuning policy                                     |
| `build_optimizer()`                              | AdamW parameter groups with discriminative learning rates                                 |
| `run_epoch()`                                    | mixed-precision training/evaluation, loss, top-1/top-5, throughput, and memory accounting |
| `evaluate_predictions()`                         | held-out inference and probability collection                                             |
| `macro_metrics()`                                | aggregate classification metrics                                                          |
| QC functions                                     | 14 executable integrity/science/reproducibility checks                                    |
| `package_final_results()`                        | manifest generation, artifact validation, ZIP creation, and archive verification          |

### Device and precision behavior

The code selects CUDA when available and enables automatic mixed precision on CUDA devices. It prefers `bfloat16` when supported; otherwise it falls back to `float16`. Gradient scaling is enabled only for the `float16` path.

The default setting is `STRICT_DETERMINISM = False`. Seeds are still fixed at **42** at the Python, NumPy, and PyTorch levels, but cuDNN benchmarking is allowed and strict deterministic algorithms are not enforced. Therefore, the experiment is reproducible at the protocol/configuration level, but it does not claim bitwise-identical outputs across all hardware and software stacks.

### Performance-oriented engineering

Several implementation details are explicitly designed for constrained Colab GPU sessions:

* metadata resolution is vectorized and avoids opening every image during indexing;
* dataset paths and labels are copied into NumPy arrays to reduce per-sample pandas overhead;
* persistent DataLoader workers are reused across multiple epochs within an experiment;
* validation and test inference use `torch.inference_mode()`;
* top-1 and top-5 metrics are obtained from one top-k computation per batch;
* metric accumulators remain on-device until aggregate values are required;
* final models are moved back to CPU between test evaluations to reduce VRAM pressure;
* unchanged ZIP integrity checks are memoized;
* the final archive is validated for CRC integrity, duplicate members, and manifest consistency.

## Installation

### Recommended environment: Google Colab

The project is designed around a Colab-style workflow with optional Google Drive caching. The notebook reports its exact runtime environment at execution time and creates the result tree under `/content/multi_species_weed_recognition`.

The source installs `timm` automatically when required. The effective package constraint is:

```bash
pip install "timm>=1.0,<2.0"
```

The surrounding environment must provide PyTorch, Torchvision, NumPy, pandas, scikit-learn, Matplotlib, Pillow, tqdm, and `requests`; the default Colab environment used for the recorded run already provided these packages.

### Dataset acquisition modes

The code supports these modes through `DATA_SOURCE_MODE`:

* `auto`: prefer validated Google Drive assets, then automatic public acquisition;
* `local`: use an existing local DeepWeeds archive or extracted dataset;
* `drive`: require a validated Google Drive archive or extracted dataset;
* `upload`: manually upload a DeepWeeds ZIP in Colab;
* `official_git`: clone the official repository for compatibility checks (the repository itself does not contain the complete image archive).

With the default `auto` behavior, the acquisition logic can use a validated Google Drive cache, the official DeepWeeds Google Drive image file, a public Kaggle mirror, or a verified Zenodo archive, while `labels.csv` is obtained from the official repository.

## Usage

The implementation is a **notebook-oriented experiment pipeline**, not a conventional command-line package. The supplied `.ipynb` is the reproducible entry point for the recorded run.

The execution flow is:

```text
Dataset acquisition
        ↓
Metadata-only label/path resolution
        ↓
Class completeness + label consistency checks
        ↓
Leakage-aware 5-fold-derived 60/20/20 split
        ↓
Model-specific transforms
        ↓
Balanced class weights from training data only
        ↓
6 pilot configurations (2 per architecture)
        ↓
Validation-based configuration selection
        ↓
Final head training + selective fine-tuning
        ↓
Best-validation checkpoint restoration
        ↓
Held-out test evaluation
        ↓
Confusion matrices + per-class metrics + error analysis
        ↓
14 executable QC passes
        ↓
Final artifact packaging
```

The notebook can be configured through the top-level experiment variables, including `SEED`, `COMPUTE_PROFILE`, `RUN_PILOTS`, `RUN_FINAL_MODELS`, `VALIDATE_ALL_IMAGES`, `STRICT_GROUPING`, and `STRICT_DETERMINISM`.

## Results and Artifact Layout

A successful run creates the following major result directories:

```text
results/
├── checkpoints/
├── configs/
├── figures/
├── logs/
├── metrics/
├── predictions/
└── tables/
```

Important artifacts include:

| Artifact                            | Purpose                                                          |
| ----------------------------------- | ---------------------------------------------------------------- |
| `tables/dataset_summary.csv`        | dataset-level summary                                            |
| `tables/class_distribution.csv`     | per-class counts                                                 |
| `tables/split_summary.csv`          | partition sizes                                                  |
| `tables/split_class_counts.csv`     | class counts by partition                                        |
| `tables/split_metadata.csv`         | complete deterministic split manifest                            |
| `configs/split_config.json`         | grouping and fold assignment policy                              |
| `configs/split_metadata.sha256`     | checksum of the split manifest                                   |
| `configs/selected_configs.json`     | selected pilot configuration per model                           |
| `configs/experiment_manifest.json`  | machine-readable experiment configuration                        |
| `configs/environment.json`          | runtime and library versions                                     |
| `metrics/*_per_class_metrics.csv`   | class-level test metrics                                         |
| `metrics/*_confusion_matrix.npy`    | raw confusion matrices                                           |
| `metrics/*_confident_errors.csv`    | highest-confidence incorrect predictions                         |
| `predictions/*_predictions.csv`     | ground truth, prediction, correctness, and predicted probability |
| `predictions/*_probabilities.npy`   | full test-set class-probability matrix                           |
| `figures/*_loss_curve.png`          | training/validation loss curves                                  |
| `figures/*_accuracy_curve.png`      | top-1 accuracy curves                                            |
| `figures/*_learning_rate_curve.png` | learning-rate schedules                                          |
| `figures/*_confusion_matrix.png`    | confusion-matrix visualizations                                  |
| `figures/*_error_montage.png`       | selected high-confidence errors                                  |
| `metrics/validation_passes.json`    | all 14 QC results                                                |
| `final_results.zip`                 | verified archive of experiment artifacts                         |

The final packaging stage excludes pilot checkpoints by default because they are intermediate artifacts; set `INCLUDE_PILOT_CHECKPOINTS_IN_PACKAGE = True` to include them.
