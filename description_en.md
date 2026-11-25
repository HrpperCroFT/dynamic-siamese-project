# Reference-conditioned kernel generation for cross-domain few-shot similarity

## 1. Task

The goal is to compare images from novel classes that were not present among the training classes. Given an image pair, the model produces a similarity score. Quality is measured by how often a randomly sampled same-class pair scores higher than a different-class pair.

The experiment uses leave-one-dataset-out (LOOCV): one dataset is held out as the target domain, while the other five are source domains for the baseline encoder. In the target domain, `K` support images per class are selected for adaptation and 10 disjoint query images per class are used for evaluation. We use `K = 5, 10, 15` and five seeds per fold.

## 2. Proposed architecture

The shared encoder is a ResNet18 trained on source data. Its parameters are frozen during target adaptation in the main comparisons. Hyper and Hyper-Sym use `layer4` features (512 × 7 × 7): two personalized 3 × 3 convolution kernels are generated from a reference image, each with 16 output channels. The kernels are applied to the second image's feature map using grouped convolution. After BatchNorm, GELU, and global average pooling, the resulting features are concatenated and an MLP produces the comparison score.

The one-directional Hyper method adds the comparison head output to cosine similarity between embeddings:

`s(a,b) = cos(e(a),e(b)) + α h(a→b)`.

The coefficient `α` is initialized to zero. Hyper-Sym averages the head outputs in both directions, `a→b` and `b→a`, while keeping the shared cosine residual. Hyper-Pure uses the learned head output without the cosine residual. Nonspatial is an ablation with one globally generated kernel.

During adaptation, Hyper comparators optimize the generator and head using support pairs: all positive within-class combinations and a comparable number of negative pairs. Training uses BCE-with-logits on `5 × similarity`, five adaptation epochs, and AdamW with `lr=3e-4`. Query images are not used for this optimization.

## 3. Compared methods

In the main LOOCV run, all methods start from the same source checkpoint and use the same support/query samples. The encoder is frozen for every method except Fine-tune.

- **Cross-domain baseline** — cosine similarity between embeddings, without target adaptation.
- **ProtoNet** — class prototypes are computed from support; query embeddings are compared with prototypes, and pair scores are similarities between posterior vectors for two images.
- **Linear** — a linear probe trained on support.
- **CosClf** — a cosine classifier trained on support.
- **RelationNet** — an MLP comparison head on image-pair embeddings, without kernel generation.
- **BN-adapt** — update BatchNorm statistics using support.
- **Fine-tune** — target adaptation of `layer4` and the embedding head of the original Siamese model.
- **RelationGCN** — task-specific support-guided adaptation, described below.
- **Nonspatial, Hyper-Pure, Hyper, Hyper-Sym** — reference-conditioned comparator variants.

### What RelationGCN means in this project

This is an adaptation of the idea of support-guided spatial feature enhancement, not an exact reproduction of the original RelationGCN or its original training protocol. A frozen ResNet18 computes `layer3` feature maps (256 channels). Local depthwise 3 × 3 kernels are generated from the reference feature map for each spatial position and channel; their responses are mixed, and the enhancement is added through a learned residual gate. Pair scoring is symmetric, and the enhancer is trained on target support while the original encoder remains frozen.

The implementation deliberately avoids globally generated dense `C × C` kernels, which would require a very large generator and substantial computation. The current RelationGCN is a separate lightweight per-position depthwise adaptation. Conclusions apply to this variant. In particular, old runs from before the implementation was corrected must not be combined with the new tables: the old ranking (including wins on Cars, Flowers, and Aircraft reported by the user) came from a different code version. In the new final runs, RelationGCN leads on Flowers, while Hyper-Sym leads on the other five folds.

## 4. Datasets and sampling protocol

Six target domains are used: CUB200, Cars196, Stanford Dogs, Oxford Flowers 102, NABirds, and FGVC Aircraft. In each fold, each of the five source domains is capped at a shared budget of 50 classes × 27 images per class: 1,350 images per domain and 6,750 images per fold. Aircraft limits the number of classes to 50, while NABirds limits the number of images per class to 27.

Each target fold uses 50 classes. This subset is fixed across seeds; support and query images are shuffled using separate seeds and do not overlap, with 10 query images per class. For NABirds, evaluation uses a fixed sample of 50 classes, although the dataset contains 550 classes in the split used here.

For pair accuracy, five positive and five negative pairs are randomly sampled per query class. All sampled positive scores are then compared with all sampled negative scores. This estimates pair accuracy from sampled pairs rather than exhaustively enumerating every possible image pair. Each table entry is the mean across five seeds; `±` is the standard deviation across seeds within a fold. Summary values across the six folds are the unweighted mean of fold-level means and fold-level standard deviations, respectively.

## 5. Main results: frozen encoder, K = 5/10/15

Each cell shows `mean across six folds ± mean seed standard deviation across folds`. The baseline is shown separately as mean cross-domain pair accuracy.

| Method | K=5 | K=10 | K=15 |
|---|---:|---:|---:|
| Cross-domain baseline | 0.7280 | 0.7282 | 0.7216 |
| ProtoNet | 0.7902 ± 0.0187 | 0.8057 ± 0.0216 | 0.8094 ± 0.0141 |
| Linear | 0.7430 ± 0.0223 | 0.7455 ± 0.0256 | 0.7409 ± 0.0243 |
| RelationNet | 0.6765 ± 0.0262 | 0.6965 ± 0.0240 | 0.6980 ± 0.0249 |
| RelationGCN (adaptation) | 0.7858 ± 0.0181 | 0.8322 ± 0.0174 | 0.8506 ± 0.0172 |
| BN-adapt | 0.7588 ± 0.0198 | 0.7510 ± 0.0217 | 0.7457 ± 0.0163 |
| Nonspatial | 0.7252 ± 0.0245 | 0.7920 ± 0.0232 | 0.8293 ± 0.0155 |
| Hyper-Pure | 0.7405 ± 0.0248 | 0.8222 ± 0.0182 | 0.8527 ± 0.0176 |
| Hyper | 0.7651 ± 0.0288 | 0.8330 ± 0.0205 | 0.8554 ± 0.0176 |
| CosClf | 0.7762 ± 0.0167 | 0.7802 ± 0.0225 | 0.7756 ± 0.0226 |
| Fine-tune | 0.7875 ± 0.0205 | 0.8324 ± 0.0128 | 0.8591 ± 0.0124 |
| **Hyper-Sym** | **0.8135 ± 0.0185** | **0.8663 ± 0.0164** | **0.8821 ± 0.0169** |

### RelationGCN and Hyper-Sym by fold

Cells show `RelationGCN / Hyper-Sym`; bold marks the higher mean. These are fresh runs with the corrected RelationGCN adaptation.

| Target fold | K=5 | K=10 | K=15 |
|---|---:|---:|---:|
| CUB | 0.8349 / **0.8477** | 0.8616 / **0.8869** | 0.8599 / **0.8958** |
| Cars | 0.6785 / **0.7539** | 0.7617 / **0.8177** | 0.7937 / **0.8268** |
| Dogs | 0.8338 / **0.8980** | 0.8519 / **0.9330** | 0.8681 / **0.9406** |
| Flowers | **0.8836** / 0.8614 | **0.9274** / 0.9245 | **0.9458** / 0.9350 |
| NABirds | 0.7604 / **0.7803** | 0.7986 / **0.8307** | 0.8065 / **0.8587** |
| Aircraft | 0.7236 / **0.7395** | 0.7918 / **0.8047** | 0.8298 / **0.8356** |

Hyper-Sym has the highest fold-averaged score at all three K values and beats RelationGCN on five of six domains. RelationGCN leads on Flowers for every K. Both spatial/hypernetwork methods improve on average as K grows; RelationNet remains substantially below the baseline, while Linear changes little.

Hyper-Sym does not improve equally on every domain. For example, at K=15 it leads RelationGCN by 7.25 percentage points on Dogs, while RelationGCN leads by 1.08 points on Flowers. These differences are descriptive and should not be treated as a strict significance test with only five seeds.

## 6. Separate branch: fine-tuning the encoder plus methods, K=10

An additional LOOCV run used `--finetune`. For each fold and seed, `layer4` and the embedding head were unfrozen; training ran for up to 15 epochs with early stopping, and 20% of support images per class were reserved for internal validation. The listed head/adaptation was applied afterward. This protocol differs from the frozen-encoder main table and should not be mixed with it.

| Head after fine-tuning | Mean pair accuracy ± mean seed std | Train time per fold/seed | Peak allocated memory* |
|---|---:|---:|---:|
| Cosine (no additional head) | 0.8116 ± 0.0164 | 16.6 s | 944 MB |
| ProtoNet | 0.8552 ± 0.0152 | 16.6 s | 485 MB |
| Linear | 0.8141 ± 0.0149 | 16.6 s | 574 MB |
| RelationNet | 0.7236 ± 0.0230 | 16.7 s | 552 MB |
| RelationGCN | 0.8511 ± 0.0116 | 72.0 s | 1,170 MB |
| Nonspatial | 0.8088 ± 0.0140 | 30.5 s | 1,288 MB |
| Hyper-Pure | 0.8145 ± 0.0181 | 32.5 s | 1,259 MB |
| Hyper | 0.8280 ± 0.0177 | 32.4 s | 1,066 MB |
| **Hyper-Sym** | **0.8577 ± 0.0101** | 36.4 s | 1,078 MB |

*Peak memory is the maximum allocated PyTorch CUDA memory, not total process or device memory use.

Best combinations by target fold: ProtoNet on CUB, Dogs, Flowers, and NABirds; Hyper-Sym on Cars; RelationGCN on Aircraft. Fine-tuning improves some methods, but Hyper-Sym's mean is lower than its frozen-encoder K=10 result (0.8577 vs. 0.8663). Additional encoder training does not provide a universal gain. The previously mentioned unsuccessful end-to-end attempt was a separate experiment and is not included in these measurements or tables.

### Training-time correction

In `cross_validate.py`, total wall time already included encoder fine-tuning, but the separate `train_time_s` column previously omitted `finetune_encoder()` and could show `0.0` for a branch without an additional trainable head. Fine-tuning time is now added to the subsequent head/adaptation time. In the new CSV, all 54 `*_train_time_s_mean` values (9 methods × 6 folds) are nonzero. Wall time still includes the whole model stage. Inference time is not profiled separately for some custom models, so use `total_time_s_mean` to compare end-to-end cost.

## 7. Controlled comparison: dense RelationGCN at K=10

To test whether the earlier RelationGCN was simply more expressive, a separate `relationgcn-full` variant was added. It reconstructs the dense version from the saved old design: a vector of reference `layer3` features generates four full `256×256×3×3` kernels; each kernel mixes channels and operates over the entire spatial field. The invalid old `conv2d` call was replaced with mathematically equivalent grouped convolution, with a separate group per example. This is a controlled reconstruction of the old idea, not an exact copy of the unknown file used for historical CSVs.

The run used the same current source checkpoints and identical support/query samples as the K=10 table above: six folds, five seeds, 10 query images per class, and 15 adapter-training epochs. Mean pair accuracies:

| Fold | Dense `relationgcn-full` | Depthwise RelationGCN | Hyper-Sym |
|---|---:|---:|---:|
| CUB | 0.8731 ± 0.0132 | 0.8616 ± 0.0170 | **0.8869 ± 0.0220** |
| Cars | **0.8228 ± 0.0126** | 0.7617 ± 0.0217 | 0.8177 ± 0.0102 |
| Dogs | 0.8841 ± 0.0194 | 0.8519 ± 0.0116 | **0.9330 ± 0.0120** |
| Flowers | **0.9519 ± 0.0096** | 0.9274 ± 0.0084 | 0.9245 ± 0.0147 |
| NABirds | **0.8359 ± 0.0175** | 0.7986 ± 0.0259 | 0.8307 ± 0.0171 |
| Aircraft | **0.8453 ± 0.0139** | 0.7918 ± 0.0195 | 0.8047 ± 0.0224 |
| **Mean across folds** | **0.8689 ± 0.0144** | 0.8322 ± 0.0174 | 0.8663 ± 0.0164 |

In this controlled comparison, the dense variant beats depthwise RelationGCN on all six folds (by 3.67 percentage points on average) and slightly exceeds Hyper-Sym on average (by 0.26 points), although Hyper-Sym remains better on CUB and Dogs. Dense RelationGCN beats Hyper-Sym on Cars, Flowers, NABirds, and Aircraft. Thus the old RelationGCN likely benefited from greater capacity; the broad claim that “RelationGCN is better overall” is not supported.

This added capacity is expensive: its generator has about 153.4M parameters versus about 0.63M for the depthwise version. On an RTX 4070 SUPER, dense RelationGCN took about 311 seconds and 3.1 GB per fold/seed on average; depthwise took about 57 seconds and 0.75 GB at K=10. Five seeds and only 50 target classes are insufficient to establish the 0.26-point advantage over Hyper-Sym as statistically convincing.

The full table is saved in `results/final_relationgcn_full_k10.csv`. It can be reproduced with `--arches relationgcn-full` and the same K=10 parameters.

## 8. Result files and reproduction

The run-by-run register, including artifacts removed during cleanup and uncertainty about historical sweeps, is maintained in [`experiments.md`](experiments.md).

Final tables and fold-level metrics, including standard deviations, time, and peak memory:

- `results/final_loocv_k5.csv`
- `results/final_loocv_k10.csv`
- `results/final_loocv_k15.csv`
- `results/final_loocv_k10_finetune.csv`
- `results/final_protonet_k10.csv` (supporting corrected ProtoNet values)
- `results/final_relationgcn_full_k10.csv`

For K=10, corrected ProtoNet fields are in `final_loocv_k10.csv`; the previous ProtoNet column incorrectly duplicated the baseline. Superseded CSVs and logs have been removed from the working directory.

All Python commands were run from the project root through `.venv\Scripts\python.exe`. The main run can be reproduced as follows:

```powershell
foreach ($K in 5, 10, 15) {
  .\.venv\Scripts\python.exe -u scripts\cross_validate.py `
    --datasets cub,cars,dogs,flowers,nabirds,aircraft `
    --arches protonet,linear,relationnet,relationgcn,bnadapt,nonspatial,hyper-pure,hyper,cosclf,finetune,hyper-sym `
    --seeds 1,2,3,4,5 --K $K --M-per-class 10 `
    --neg-same-domain-ratio 0.5 --lr-side 3e-4 --adapt-epochs 5 `
    --feature-level l4 --n-kernels 2 --out-ch 16 `
    --output "results/final_loocv_k$K.csv"
}
```

The separate fine-tuning combination used the same datasets, seeds, and K=10, with default fine-tuning parameters and `--finetune`; its results are in `results/final_loocv_k10_finetune.csv`.

**Baseline encoder source.** In these runs, `cross_validate.py` loaded existing `baseline_loo_<fold>.pt` cache files rather than retraining six source encoders. Logs from the earlier main run show a 50 × 27 budget and checkpoint writes. The checkpoints themselves contain only `tower` and `head` weights, with no training configuration or metadata, so their complete provenance cannot be independently verified. Seeds in the new CSVs measure support/query sampling and adaptation variance with these fixed encoders; they are not five independent ResNet training runs. A fully fresh replication must bypass or remove these cache files with `--skip-baseline-cache` and rerun every fold and K.

## 9. Reliability of conclusions

The tables now agree on fold count, K, query-set size, and number of seeds; ProtoNet values and fine-tuning time have been corrected. They are more reliable than the original values in the old `loocv_k*.csv` files, but remain research results rather than definitive evidence of superiority:

1. Five seeds vary sampling and adaptation, but the encoder checkpoint is fixed; variability from training the source encoder was not measured.
2. Pair accuracy is based on a small number of random pairs per class rather than exhaustive pair enumeration, adding sampling noise.
3. Evaluation uses a fixed subset of 50 target classes per dataset. NABirds therefore uses only part of the available classes.
4. RelationGCN is our task-adapted architecture, not an exact paper reproduction; results from the earlier implementation must not be directly compared.
5. Baseline caches contain weights without full provenance. The strictest replication requires retraining the encoders.

## 10. Related works

### Conditional generation of convolutional filters

- **Dynamic Filter Networks** — De Brabandere et al. introduce filters generated dynamically as a function of the input. This is the foundational idea closest to Hyper, where features from a reference image determine convolution kernels for processing the second image: [NeurIPS 2016 paper](https://papers.neurips.cc/paper_files/paper/2016/file/8bf1211fd4b7b94528899de0a43b9fb3-Paper.pdf).
- **HyperNetworks** — Ha, Dai, and Le describe one neural network generating parameters for another. This is a general source for hypernetworks, not a description of this project's specific architecture: [ICLR 2017 paper](https://openreview.net/pdf?id=rkpACe1lx).
- **Learning Instance and Task-Aware Dynamic Kernels for Few-Shot Learning** — Ma et al. use generated dynamic kernels in few-shot tasks. It is a close problem-level example, but their model forms instance- and task-aware kernels and differs from the Hyper variant used here: [arXiv:2112.03494](https://arxiv.org/abs/2112.03494).

### Pair comparison and few-shot learning

- **Learning to Compare: Relation Network for Few-Shot Learning** — Sung et al. train a deep function to compare example pairs. This is the source associated with the RelationNet baseline in the experiment tables; it is not an exact source for Hyper or RelationGCN: [CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Sung_Learning_to_Compare_CVPR_2018_paper.html).
- **Prototypical Networks for Few-shot Learning** — Snell, Swersky, and Zemel define the prototype baseline, representing each class by the center of its support examples in feature space: [NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html).
- **Few-Shot Learning with Graph Neural Networks** — Garcia and Bruna formulate few-shot inference as message passing over a support/query graph. This is relevant graph-based context, although the current RelationGCN adaptation does not build such a graph: [ICLR 2018](https://openreview.net/pdf?id=BJj6qGbRW).

### RelationGCN and adaptation scope

- **Few Shot Object Detection for SAR Images via Feature Enhancement and Dynamic Relationship Modeling** — Chen et al. introduce RelationGCN for few-shot detection in SAR images, combining support attention, support-guided query feature enhancement, and a graph-based relationship module. In this project, RelationGCN names an adaptation of that line of ideas for pairwise similarity on natural images. The current code uses local depthwise kernels and symmetric cosine comparison, rather than Faster R-CNN and the original graph module; it is therefore not a reproduction of the paper: [Remote Sensing, 2022, 14(15), 3669](https://www.mdpi.com/2072-4292/14/15/3669), [DOI: 10.3390/rs14153669](https://doi.org/10.3390/rs14153669).
- The RelationGCN paper cites **Semi-Supervised Classification with Graph Convolutional Networks** by Kipf and Welling for its base graph operator: [arXiv:1609.02907](https://arxiv.org/abs/1609.02907). That graph operator is not directly implemented in the current depthwise adapter.

### Encoder and datasets

- ResNet18 is based on **Deep Residual Learning for Image Recognition** by He et al., CVPR 2016: [CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).
- Datasets: [CUB-200-2011](https://authors.library.caltech.edu/records/cvm3y-5hh21/files/CUB_200_2011.pdf) (Wah et al., 2011); [Stanford Cars](https://www.cs.princeton.edu/~jiadeng/paper/KrauseStarkDengFei-Fei_3DRR13.pdf) (Krause et al., 2013); [Stanford Dogs](https://people.csail.mit.edu/khosla/papers/fgvc2011.pdf) (Khosla et al., 2011); [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/publications/2008/Nilsback08/nilsback08.pdf) (Nilsback & Zisserman, 2008); [NABirds](https://openaccess.thecvf.com/content_cvpr_2015/html/Horn_Building_a_Bird_2015_CVPR_paper.html) (Van Horn et al., CVPR 2015); [FGVC-Aircraft](https://arxiv.org/abs/1306.5151) (Maji et al., 2013).

These citations identify related ideas and the sources of baseline methods. They do not imply that the project variants exactly reproduce the original architectures or protocols.
