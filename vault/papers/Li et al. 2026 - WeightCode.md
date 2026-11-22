---
type: paper
title: "WeightCode: Installing New Capabilities into Neural Networks via Weight-Space Programming"
short: "Li et al., 2026"
authors: [Bo Li, Taoyong Cui, Bob Zhang, Mengran Li, Chengyang Zhang, Yang Zhang, Pheng Ann Heng, Wanli Ouyang, Zhenfei Yin, Yingcheng Wu, Ling Yang]
year: 2026
venue: "arXiv:2611.25476v1 [cs.LG], 19 Nov 2026"
url: "https://arxiv.org/abs/2611.25476"
local_text: "papers/pdf/text/arxiv_2611.25476.txt"
tags: [paper, weight-space-programming, weight-editing, model-editing, gradient-free, capability-installation]
read_status: full-text-read
---

# Li et al., 2026 - WeightCode

## TL;DR
A frozen LLM (Claude Opus 5.5 at maximum reasoning effort) writes an executable "weight program" $p$ that reads a trained network's weights $W$ and task knowledge $I$ and returns edited weights $W' = p(W, I)$ without gradient-based training. Candidates are selected with 8 labeled checks under a retention rule that bounds the loss on old behavior. Headline numbers: SCAN add-primitive exact match 0.0% to 99.8% with 257 changed weights; BabyAI pickup success 0.0% to 98.4% from 8 demonstrations; four ImageNet ResNet-18 programs reach net gains of +8.84 to +17.62 points under corruption versus +8.67 / +3.38 for the best of eight fine-tuning methods. Rerunning a kept program on five independently trained ViT-Lites keeps 83% to 110% of its source gain, while copying the fixed weight edit gains -1.42 to +0.01 points on average. All networks are small (at most 11.7M weights); no large pretrained LLM is edited.

## Problem
In the paper's framing: learning a capability during training and installing one into an already trained network are distinct problems. Installation requires "translating task knowledge into new computations while preserving previously acquired capabilities". The paper asks four questions about Weight-Space Programming (WSP, a term it takes from [[Takehi et al. 2026 - Weight-Space Programming]] and from Li et al., WeightLoop, arXiv:2611.07319):
- Installation: can a weight program install a new capability while preserving old ones?
- Mechanism: what computation does it install, and how does it produce the capability?
- Reuse: can the same program be rerun without the LLM on independently trained networks?
- Succession: can a new program extend an already edited network while keeping earlier installed capabilities?

## Method (step by step)

### Definitions
1. A weight program $p$ is code that takes a trained network's weights $W$ and task knowledge $I$ and returns new weights (Eq. 1):
   $$W' = p(W, I), \qquad \Delta W = p(W, I) - W.$$
   $\Delta W$ depends on the network the program reads, so the same program run on another network computes another edit.
2. A program "may read every weight, run the network's forward pass on inputs and read $I$".
3. No architecture growth: "The updated network keeps its architecture and forward code, with only its weights changed, and runs on its own at inference." New computation is written into existing, little-used or dead units, existing rows, or existing attention heads (see "What programs write").

### What the LLM is given (task knowledge $I$), by level
- Labeled examples (image classifiers): 64 labeled images of the new condition (16 that the network misclassifies and 48 drawn at random, with the network's predictions) and 64 labeled clean images.
- BabyAI: 8 demonstrations of the new mission (each step's observation, instruction and expert action) and 64 old steps, with the policy's action logits.
- Condition labels withheld: same condition images and the network's predictions, but not their labels and no statement of how they were chosen; clean images and checks keep labels.
- A rule: SCAN's published interpretation grammar with 64 old commands (with answers and the network's predictions) and, on the add-primitive split, the training row of the new primitive. For the ViT: the formula that generates the corruption and 64 labeled clean images, with no image of the condition.
- A capability's name: one sentence asking for robustness to additive Gaussian pixel noise, with no image, label or prediction; given with the 8 checks or with none.
- Existing weights: the kernel holds $W$ and the forward pass. In a control, the same LLM gets parameter names and shapes, forward code, examples and checks "but no weight value or network output".

### How the program is written and verified
4. Writer: "Programs are written by Claude Opus 5.5 at maximum reasoning effort." One LLM session per program, on one trained network.
5. Workspace: a Python kernel that holds $W$, the forward pass and $I$; at most 150 cells; 90 to 120 minutes (140 on ImageNet, 190 on the CIFAR-10 ResNet-18). Sessions take 30 to 130 minutes; CIFAR-10, SCAN and BabyAI sessions cost US$5 to 13 at API prices (Table 7: US$7.12 to 13.08 per ViT-Lite session).
6. Loop: "Program construction alternates between inspecting the network, writing candidates and evaluating them."
7. Check: runs a candidate on the network and returns two counts on a check split that $I$ does not contain: $S_{new}$ (new-condition inputs the candidate fixes minus those it breaks) and $S_{old}$ (same count on old inputs). A session has 8 checks.
8. Retention rule: keep the checked candidate with the highest $S_{new} + S_{old}$ among those with $S_{old} \ge -20$ (a loss of at most 0.5% of 4,000 clean check images). SCAN bound: 5 of 1,000 old commands. BabyAI bound: 3 of 500 old episodes. If no candidate qualifies, the network is left unchanged.
9. Constraints on the program run: sandbox with NumPy only, no network access, 300 s limit; at most 20% of the weights may change; the change to each tensor of rank two or more is at most half of that tensor's Frobenius norm (applies to the image classifiers, not to SCAN or BabyAI, whose edits write whole embedding or output rows).

### Gradients and feedback (explicit)
- No gradients in the main protocol. Allowed: "weight and activation statistics, SVD and PCA, label-free closed-form fits and single-unit ablations", forward evaluations and closed-form computations.
- Excluded: "Gradient-based and finite-difference optimization of objectives computed from the network's outputs (by automatic differentiation, hand-written backpropagation or finite differences), solvers on label-derived targets and iterated label-driven updates".
- Three practices allowed inside the protocol: one overall edit strength found by a search against a label-free bound on the change of clean outputs; a one-shot closed-form bias set from the network's own logit margins (BabyAI); a head bias set so that the edited network's mean logits on label-free inputs equal the original's.
- Labeled feedback is used to evaluate and select, not to fit values: a single-round session scores at most 8 candidate edits against the labeled examples and receives 8 checks. Label-scored work that changes no weight (output offsets, nearest-class-mean probes) and selection inside a program's own run are counted separately. This counting rule "was set on 2026-11-06, after two sessions had been read".
- An automatic audit flags gradients, solvers on label-derived targets, iterated label-driven updates and the number of settings scored against labels.

### What programs write (how the computation is installed; Table 9)
| Network, condition, information | What the program writes | Weights changed |
|---|---|---|
| ViT-Lite, Gaussian noise, labeled | frees two little-used MLP units in each of blocks 2, 4 and 6 and rewrites them as ramp gates that subtract the measured noise offset from tokens above the clean range | 3,846 |
| ViT-Lite, Gaussian noise, condition labels withheld | estimates noise strength from excess neighbor-difference energy; chains saturating gates in the least-used MLP units of blocks 2 to 4 and the last block, each triggered by a flag the previous gate writes | 19,724 |
| ViT-Lite, Gaussian noise, name only | gives the 16 block-0 MLP units most driven by noise twin units that cancel the mean the noise induces through the GELU | 24,848 |
| ViT-Lite, contrast reduction, labeled | raises the zero-mean part of the patch-embedding filters by a factor of 1.45 and adds a gated rank-6 correction in eight freed units of block 2 | 16,648 |
| ResNet-18 (CIFAR-10), Gaussian noise, labeled | estimates noise level from high-frequency energy, applies Wiener shrinkage to each convolution in forward order, shifts following BatchNorm means; no gate | 1,718,057 |
| ResNet-18 (CIFAR-10), contrast, labeled | raises the zero-mean part of the stem convolution by 35% and adds gates on a whitened low-contrast direction in two late residual blocks | 37,192 |
| ResNet-18 (ImageNet), Gaussian noise, labeled | estimates noise from pixel statistics, synthesizes noisy copies of the clean images, removes the noisy-minus-clean activation shift at the inputs of two residual stages and of the classifier, folded into weights and BatchNorm statistics | 2,152,928 |
| ResNet-18 (ImageNet), contrast, labeled | measures the contrast factor from pixel statistics; turns the two least-used units of several residual blocks into gates that open on low-contrast inputs and add the restoring activation shift | 33,176 |
| BabyAI policy, pick up, 8 demonstrations | copies the embeddings of go and to into pick and up; builds from dead units an exact detector for a faced cell of the named color and type that triggers pickup; adds a second detector that stops the policy pressing forward into a blocking object | 21,482 |
| SCAN Transformer, add primitive, rule | moves the embedding of jump to the centroid of those of walk, look and run and adds an output row and bias for its action | 257 |
| SCAN Transformer, around right, rule | rank-one edits of attention values in blocks 2 to 4 along the network's own right-minus-left turn direction | 25,056 |

Worked example (ViT-Lite Gaussian-noise gate, Sec. 4.2 and Fig. 3a):
1. Measure the noise shift of patch tokens in the early residual stream on synthetic noisy copies of the clean images (the LLM infers additive white Gaussian noise of standard deviation about 25/255; the generating value is 0.10).
2. Free two little-used MLP units in block 2 "by folding their mean output into the bias".
3. Rewrite them as a ramp gate (Eq. 2):
   $$g(z) = \mathrm{GELU}\big(k(d^\top z - t)\big) - \mathrm{GELU}\big(k(d^\top z - t) - K\big)$$
   which subtracts the measured offset only from tokens whose projection $d^\top z$ on a whitened noise direction lies above the clean range.
4. Code excerpt shown in Fig. 1a / 3a: `j1, j2 = free[0], free[1]`; `W1[j1] = kappa * d`; `b1[j1] = -kappa * t`; `W2[:, j1] = -off / K_RAMP`. (inference: the second unit `j2` carries the shifted GELU term with the opposite output sign; the figure does not print those lines.)
5. Two gates of the same form act on the class token in blocks 4 and 6.

### Copy, rerun, successive update (Sec. 3.2)
- Copying: add the edit $W_1 - W_0$ unchanged to another network.
- Rerunning: execute the code of $p_1$ unchanged, with no further LLM session, on the target network with that network's own task examples. Takes "a few minutes with no LLM call".
- Successive update: ask the LLM for a new program $p_2$ for the edited network, $W_2 = p_2(W_1, I_2)$, given the edited network, the code of $p_1$, labeled examples of the new condition, and checks that also report every earlier condition under the retention rule for each.

### Data splits and criteria
- Per condition, three disjoint held-out splits: evidence split (2,000 images; source of the condition images in $I$), check split (4,000), selection split (4,000; plays no part in any session, check, retention rule or fine-tuning choice). Clean images of $I$ are one fixed draw from the network's training images.
- Test sets: official CIFAR-10 and CIFAR-10-C test files; 38,000 ImageNet validation images (12,000 of the 50,000 form the three splits and the clean pool); the third of SCAN's test commands kept for final evaluation; held-out BabyAI episodes. "Each kept network was read on them once, after every selection was final."
- Metrics: gain = accuracy change on the condition; clean change = accuracy change on clean images; net gain = their sum.
- Prespecified success criterion: gain of at least 10 points (ImageNet: 7.0 and 8.5 points), clean change of at least -1.0, net gain at least one point above the strongest reference that obeys the program limits.

## Experiments and numbers

### Networks
| Network | Size | Clean accuracy |
|---|---|---|
| ViT-Lite-7/4 (CIFAR-10, 40,000 training images) | 3.7M weights | 89.35% |
| ResNet-20 (CIFAR-10) | 0.27M weights | 91.28% |
| ResNet-18, torchvision ImageNet, unchanged | 11.7M weights | 69.70% (on the paper's ImageNet test images) |
| ResNet-18 fine-tuned to CIFAR-10 | 11.2M weights | 95.78% |
| SCAN Transformer (per split) | width 128, 4 layers; weight count not stated | not stated |
| BabyAI policy (feedforward, FiLM conditioning; imitation on 50,000 expert go-to demonstrations) | not stated | not stated |
| Synthetic sequence Transformer (Sec. A.10.1) | width 64, 205,008 weights | 100% on old operations |

Corruptions: CIFAR-10-C at severity 5; ImageNet-C Gaussian noise at severity 3 and contrast at severity 4. On the ViT-Lite, Gaussian noise leaves 35.55% accuracy and contrast reduction 58.88%.

Baselines: eight fine-tuning methods (full, last layer, LayerNorm only / BatchNorm on ResNet-18, LoRA rank 4 and 16, full without clean images, full with LwF, full with L2-SP), 8 configurations each, same examples, same 8 checks, same retention rule; eleven fixed references (label fits and label-free rules including Tent, BN adaptation, SAR).

### Rules and demonstrations (Table 2, held-out test sets)
| Setting | New, base | New, after | Old (change, pt) | Weights | Other route | Other new | Other old |
|---|---|---|---|---|---|---|---|
| SCAN, add primitive | 0.0% | 99.8% | 0.0 | 257 | grammar-generated fine-tuning (152,718 entries) | 99.6% | 0.0 |
| SCAN, around right | 17.6% | 89.6% | -0.3 | 25,056 | grammar-generated fine-tuning (fixed 19% of each tensor) | 100.0% | -0.1 |
| BabyAI, pick up | 0.0% | 98.4% | +0.4 | 21,482 | behavior cloning, embeddings and head | 1.1% | -0.6 |

- Fine-tuning on the given rows alone keeps at most +1.21 (add primitive) and +2.61 (around right) points of net gain (selection split).
- BabyAI: success remains 98.9% on episodes whose object color and type never appear in the demonstrations (selection split).
- SCAN add primitive: embedding change alone installs nothing; output change alone reaches 26.90% exact match; together 99.84% (selection split).

### Matched comparison with fine-tuning (Table 3, held-out test sets, points)
| Network | Condition | Program gain | Clean | Net gain | FT best of 8, kept | FT no bound (clean) | Fixed refs kept | Weights changed |
|---|---|---|---|---|---|---|---|---|
| ResNet-18 ImageNet | Gaussian noise | +10.72 | -0.25 | +10.47 | +8.67 | - | +8.84 | 2,152,928 |
| ResNet-18 ImageNet | contrast | +9.31 | -0.47 | +8.84 | +3.38 | - | +6.00 | 33,176 |
| ViT-Lite | Gaussian noise | +32.68 | +0.11 | +32.79 | +22.78 | +29.11 (-1.98) | +16.48 | 3,846 |
| ViT-Lite | contrast | +17.45 | -0.94 | +16.51 | +7.89 | +7.89 (-0.34) | +8.25 | 16,648 |
| ResNet-18 CIFAR-10 | Gaussian noise | +35.61 | -0.38 | +35.23 | +9.47 | +30.18 (-2.15) | +9.22 | 1,718,057 |
| ResNet-18 CIFAR-10 | contrast | +21.43 | -0.15 | +21.28 | +10.01 | +26.70 (-2.26) | +13.51 | 37,192 |

- ImageNet second sessions (Table 8): Gaussian noise +18.02 gain, -0.40 clean, +17.62 net; contrast +14.48, -0.22, +14.26. ImageNetV2 clean change of the four programs: -0.34 to -0.89 points.
- Edit sizes range "from 3,846 modified weights for the ViT-Lite noise gate to 2,152,928 for the ImageNet noise correction (0.10% to 18% of each network)".

### Every ViT-Lite session (Table 7, held-out test; copied = selection split)
| Condition | Network / session | Gain | Clean | Net | FT kept | Criterion | Checks | US$ | Rerun share | Copied edit |
|---|---|---|---|---|---|---|---|---|---|---|
| Gaussian | seed 100, 1 | +32.68 | +0.11 | +32.79 | +22.78 | met | 6 | 9.92 | 0.87 | -0.10 |
| Gaussian | seed 100, 2 | +12.73 | -0.53 | +12.20 | +22.78 | missed | 8 | 8.36 | 0.98 | +0.01 |
| Gaussian | seed 100, 3 | +30.80 | +0.03 | +30.83 | +22.78 | met | 8 | 8.75 | 0.91 | -0.16 |
| Gaussian | seed 101 | +30.85 | -0.27 | +30.58 | +16.31 | met | 5 | 7.81 | 0.85 | -0.34 |
| Gaussian | seed 102 | +18.35 | -0.38 | +17.97 | +10.32 | missed | 6 | 8.74 | 0.98 | -0.45 |
| contrast | seed 100, 1 | +17.45 | -0.94 | +16.51 | +7.89 | met | 8 | 9.63 | 0.92 | -0.56 |
| contrast | seed 100, 2 | +9.51 | -0.38 | +9.13 | +7.89 | missed | 6 | 8.90 | 0.96 | -0.31 |
| contrast | seed 100, 3 | +11.39 | -0.65 | +10.74 | +7.89 | met | 8 | 13.08 | 0.83 | -1.42 |
| contrast | seed 101 | +6.77 | -0.16 | +6.61 | +5.43 | missed | 8 | 10.33 | 1.10 | -0.24 |
| contrast | seed 102 | +8.45 | -0.52 | +7.93 | +9.52 | missed | 5 | 7.12 | 0.84 | -0.06 |

Summary stated by the paper: of ten programs, eight exceed the best kept fine-tuning on their own network, five meet the prespecified criterion, all ten keep 0.83 to 1.10 of their gain when rerun.

### Ablations and controls (Table 10, ViT-Lite, selection split, net gain in points)
| Variant | Gaussian | Contrast |
|---|---|---|
| Weight program | +31.75 | +17.98 |
| block-2 gate removed | +10.05 | - |
| gates held open (clean -5.03; worst class -15.5) | +19.43 | - |
| all three gates removed | returns the network to its original accuracy (Fig. 3c shows 0.05) | - |
| two random units of the block (mean of 5) | +31.86 | - |
| gate from the plain mean difference | +31.13 | +5.40 |
| unconditional feature alignment | +7.18 | - |
| gradient on the program's entries, from the original values | +3.60 | +0.72 |
| gradient on the program's entries, from the program's values | +33.35 | +18.00 |
| the LLM without weight values | +10.78 | +11.18 |
| generic gated search, no LLM (post hoc) | +24.12 | unchanged |

- Gate selectivity: the block-2 gate opens on 99.5% of noisy and 0.5% of clean patch tokens; mean noisy-minus-clean offset cut from 5.29 to 0.50; clean patch tokens move by 0.05 on average against 5.02 on noisy ones.
- Contrast program parts: first-layer gain alone +10.95, gated correction alone +9.63, together +17.93; gate opens on 99.7% of low-contrast and 0.3% of clean patch tokens.
- Data budget: on the ViT-Lite under Gaussian noise, full fine-tuning first matches the program's net gain within the retention rule at 4,096 labeled corrupted images and as many clean ones, versus 64 of each for the program (selection split).
- Less information (ViT-Lite, Gaussian noise, selection split, bar +17.075): labeled +31.75; condition labels withheld +33.37 (test: +34.43 at clean -0.04); name and 8 checks +9.43; name only, no check +15.10. Test-time adaptation on the same unlabeled images: Tent leaves the network unchanged, SAR +0.025.

### Reuse (Table 11, held-out test, five other seeds)
| Condition | Reused | Mean gain | Share | Clean |
|---|---|---|---|---|
| Gaussian | program | +28.57 | 0.87 | -0.19 to +0.07 |
| Gaussian | FT recipe | +20.79 | 0.89 | -0.85 to -0.31 |
| Gaussian | FT re-selected | +10.44 | 0.45 | >= -1.0 |
| Contrast | program | +16.04 | 0.92 | -1.10 to -0.47 |
| Contrast | FT recipe | +8.09 | 0.98 | -0.55 to -0.15 |
| Contrast | FT re-selected | +6.57 | 0.80 | >= -1.0 |

- Copied edits on independently trained ViT-Lites: -1.42 to +0.01 points on average, 7.6 to 28.2 below rerunning (selection split). Unit-by-unit activation alignment on 2,048 clean images recovers -0.01 and +4.17.
- On five ResNet-18s fine-tuned from the same ImageNet weights, copied edits carry 96% and 87% of the rerun gain (selection split).
- Clean accuracy stays within 1 point on 48 of the 50 transfers.

### Succession (Table 13)
- Noise first, then six contrast programs: +5.23 to +14.50 points of contrast robustness on held-out test; installed noise robustness moves by -0.17 to +0.23; clean accuracy within 1.47 points of the original; four of six exceed the 10-point threshold.
- Contrast first, then six noise programs: +30.9 to +37.7 points of noise robustness while contrast moves by -0.28 to +0.03 (selection split).
- Counterfactual: the contrast program written for the original network, rerun on the noise-programmed network, frees the two block-2 gate units and breaks 1,224 of 4,000 noisy check images (noise -26.6 pt); freeing those units accounts for 81% of its noise loss.

### Beyond the network's identified computation (Sec. A.10, one session each, selection split)
- Synthetic sequence Transformer: new operation installed with 100% exact match on lengths 4 to 8 and 92.75% on lengths 9 to 12, changing 4,465 weights (2.18%), in one check. Composed with an old operation the lookup reaches 5.95% exact match, whereas fine-tuning on rows generated from the same rule reaches 100% on both lengths and on the composition.
- Few-shot class-incremental CIFAR-100 (CEC protocol, ResNet-20 with cosine head): program reaches 17.20% on new classes and 64.55% class-balanced on all seen classes, against 21.70% and 65.11% for imprinted prototypes and a criterion of 26.70% and 65.11%. The criterion was not reached.

### Exploratory, outside the main protocol (Sec. A.11)
- Three sessions sized edits from finite differences of label-free output quantities. Noise-formula session: 34,304 weights, net gain +18.27 at clean -0.46 (test).
- Four-round ResNet-20 stream (Gaussian, defocus, snow, contrast): stream score after round 4 of 8.95 versus 5.15 for the best fine-tuning stream (Table 14); mean gain +10.61 at clean -1.66.

## What it establishes
- For small vision, sequence and policy networks, an LLM-written, gradient-free program can add a measured capability while a retention rule bounds loss on old behavior (Tables 2, 3, 7, 8).
- Under matched examples, checks and retention rule, the reported programs beat the best of eight evaluated fine-tuning methods on ImageNet (4 of 4 sessions) and on 10 of 12 CIFAR-10 sessions.
- The installed gates are causally responsible for the ViT-Lite noise gain (removal and held-open interventions, Table 10).
- Access to weight values matters in this setting: +31.75 / +17.98 with weights versus +10.78 / +11.18 without (one session per condition).
- Rerunning a program transfers across independently trained networks; copying the numerical edit does not, unless networks share an initialization.

## What it does NOT establish
- Nothing about large pretrained LLMs. The largest edited network has 11.7M weights (ImageNet ResNet-18); the only language-like models are a width-128 4-layer SCAN Transformer and a 205,008-weight synthetic sequence Transformer. LLMs appear only as "natural next steps".
- It does not show that programs can create representations the network lacks: the main results "build each capability on computation the network already has", and both Sec. A.10 probes fall short (5.95% composition; criterion missed on CIFAR-100).
- It is not gradient-free end to end in the sense of "no labels": labeled checks select candidates, and the fine-tuning comparison is matched on examples and checks, not on the LLM's pretrained prior ("The methods differ in how they construct updates, including the pretrained prior of the LLM that writes each program").
- Robustness of the method across LLM writers is not tested (one writer model).
- Half of the ViT-Lite sessions miss the prespecified criterion; session-to-session variance is large (for example +12.20 versus +32.79 on the same network).
- With a relaxed clean-loss bound, fine-tuning overtakes the program in some cells (for example +26.70 at clean -2.26 versus +21.28 on the CIFAR-10 ResNet-18 contrast).

## Limitations stated by the authors
The paper has no dedicated limitations section; the following caveats are stated in the text:
- Three exploratory sessions used finite differences and "fall outside the main protocol"; "The automatic audit ... did not detect these operations; they were identified by subsequent code inspection."
- The label-counting rule was set after two sessions had been read; for the first Gaussian-noise session an expanded reference grid "was fixed after the session began but before its first check".
- Grammar-generated fine-tuning includes some test commands, "so it is a rule-to-data reference rather than a held-out supervised baseline".
- One ImageNet contrast session has a clean change of -1.13 on the selection split, past the -1.0 bound.
- The first contrast program loses more than 1 point of clean accuracy on two of five networks when rerun.
- The ResNet-20 round-1 program "reads the wall clock and stops its sweep over units after 220 s"; "a full sweep produces a different $W_1$".
- In the CIFAR-100 session "the generated programs stay within a prototype-based update family and do not reach the criterion".
- Extending foundation models, including LLMs, is future work.

## Implementation notes for med_wsp
- Code: https://github.com/Gen-Verse/WeightCode (stated on the title page; repository contents not checked for this note).
- Writer model and budget: Claude Opus 5.5, maximum reasoning effort, at most 150 kernel cells, 8 checks, 90 to 120 minutes, US$5 to 13 per session. Training hardware: not stated.
- Reusable protocol pieces (independent of network size): the three-split design (evidence / check / selection) plus a once-read test set; $S_{new}$ / $S_{old}$ fixed-minus-broken counts; a hard retention bound; a fixed number of checks; the 20% weight cap and the half-Frobenius per-tensor cap; the controls of Table 10 (random units, plain mean difference, gradient on the same entries, LLM without weight values, generic search without LLM).
- 36 GB Apple M4 Max: every network in the paper is at most 11.7M weights and NumPy-only programs run in under 300 s, so the experiments would fit in memory (inference; the paper does not state hardware). Nothing in the paper tells us whether the gate construction scales to a 3.8B-parameter transformer.
- Gotcha for LLM use: the gate recipe assumes "little-used" MLP units that can be freed by folding their mean output into a bias. ViT-Lite MLP layers have biases; whether Phi-3.5 MLP layers offer the same freedom is not addressed by the paper (inference: needs checking against the Phi-3.5 architecture).

## Relevance to med_wsp (provisional)
- Provisional idea: borrow the acceptance protocol (check split, retention bound on non-medical or general behavior, fixed number of checks, single read of the MedQA test split) for our targeted edits of Phi-3.5.
- Provisional idea: the copy-versus-rerun result bears on transferring computations from [[Corbeil et al. 2025 - MediPhi]] into Phi-3.5. The paper finds copied edits work "where networks share coordinates" (shared pretrained initialization: 96% and 87% of the rerun gain). MediPhi is a fine-tune of Phi-3.5, so the two plausibly share coordinates (inference; not tested by this paper).
- Provisional idea: for failure-guided editing without a donor, the "measure a shift on a small evidence set, write a gated correction into low-use units" pattern is a candidate edit form; compare with the direction-removal edit of [[Takehi et al. 2026 - Weight-Space Programming]] experiment 8.
- Difference to keep in mind: WeightCode retains the program as the artifact and reruns it; our replication of the MIT experiments uses the turn-by-turn harness described in [[MIT replication spec (exp 3, 7, 8)]] and [[Weight-space programming]].
- Localization tools such as [[Activation patching]], [[Transcoders]] and [[Attribution graphs]] are not used in this paper; its programs rely on activation statistics, SVD / PCA and single-unit ablations.

## Open questions
- Does any of the gate or direction constructions work on a multi-billion-parameter LLM, where "little-used units" and clean / condition splits are harder to define?
- How much of the gain comes from the LLM's prior about known corruption fixes (Wiener shrinkage, contrast gain) rather than from reading weights? The "LLM without weight values" control still reaches +10.78 / +11.18.
- How sensitive are results to the writer model and to the number of checks (only 8 tested)?
- What is the analogue of "clean accuracy" for a medical QA edit: general-domain QA, MedQA items the base model already answers correctly, or both?

## Ledger card
**Full citation:** Bo Li, Taoyong Cui, Bob Zhang, Mengran Li, Chengyang Zhang, Yang Zhang, Pheng Ann Heng, Wanli Ouyang, Zhenfei Yin, Yingcheng Wu, Ling Yang. "WeightCode: Installing New Capabilities into Neural Networks via Weight-Space Programming." arXiv:2611.25476v1 [cs.LG], 19 Nov 2026.
**Source / DOI / URL:** https://arxiv.org/abs/2611.25476 ; code https://github.com/Gen-Verse/WeightCode ; local PDF `papers/pdf/arxiv_2611.25476.pdf` (26 pages).
**Why I read it:** closest paper-form treatment of weight-space programming as a method for installing capabilities; needed to know whether it offers a technique usable on Phi-3.5.
**Problem addressed:** installing a new capability into a trained network from rules, demonstrations or a few labeled examples, without gradient training, while keeping old behavior.
**Method / representation:** frozen LLM writes an executable NumPy weight program $W' = p(W, I)$; candidates selected by 8 labeled checks under a retention rule; program kept and rerun on other networks.
**Assumptions:** the capability can be connected to computation the network already performs; a check split with labels exists; low-use capacity exists to host new gates; edits limited to 20% of weights.
**Evidence used by the paper:** experiments: SCAN (2 splits), BabyAI, CIFAR-10-C on ViT-Lite / ResNet-18 / ResNet-20, ImageNet-C on ResNet-18, reuse on 5 seeds, 12 second-round sessions / analysis: causal interventions on gates, alternative constructions, cross-corruption effects / datasets or systems: SCAN, BabyAI, CIFAR-10, CIFAR-10-C, ImageNet, ImageNet-C, ImageNetV2, CIFAR-100 (CEC protocol).
**Safe citation claims:** "A frozen LLM can write gradient-free weight programs that install robustness or new behaviors in small trained networks while a retention rule bounds loss of old behavior (Li et al. 2026)."; "Rerunning a weight program on independently trained ViT-Lites kept 83% to 110% of its source gain, whereas copying the fixed edit gave -1.42 to +0.01 points on average (Li et al. 2026)."; "On SCAN add-primitive a 257-weight edit raised exact match from 0.0% to 99.8% (Li et al. 2026)."
**Unsafe / unsupported citation uses:** "WeightCode edits large language models" (it does not); "weight programs outperform fine-tuning in general" (only under matched 64 + 64 examples, 8 checks and a tight retention bound; fine-tuning overtakes with more data or a looser bound); "weight programs can add capabilities the network has no representation for" (Sec. A.10 shows the opposite tendency).
**Relation to our project:** closest prior art | overlapping method (protocol design); background for scale (no LLM-scale evidence).
**Verification status:** full text read: yes; key claim checked: yes (numbers checked against Tables 2, 3, 7, 8, 10, 11, 13 in the extracted text; Eq. 2 checked against PDF page 6).

## Links
- [[Takehi et al. 2026 - Weight-Space Programming]] - the blog post this paper cites as concurrent work and source of the term
- [[Weight-space programming]] - concept note, including how WeightCode differs from the blog's harness
- [[MIT replication spec (exp 3, 7, 8)]]
- [[Corbeil et al. 2025 - MediPhi]]
- [[Activation patching]], [[Transcoders]], [[Attribution graphs]]
