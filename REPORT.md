# Text Prompt + SAM3 vs YOLO + SAM2 for Swimming Pool Segmentation in Aerial Images

Prepared on 2026-03-29

## Executive Summary

This study compares two segmentation strategies on aerial imagery for a single target class, `swimming pool`:

- **Pipeline A**: `YOLOv8n detector -> bounding boxes -> SAM2.1_b`
- **Pipeline B**: `SAM3` with the fixed text prompt `swimming pool`

On the `iSAID -> swimming pool only` validation subset, evaluated with **union-mask IoU** on **positive images only**, Pipeline A achieved a higher mean IoU than Pipeline B:

| Metric | Pipeline A: YOLO + SAM2 | Pipeline B: SAM3 text prompt |
|---|---:|---:|
| Mean IoU | **0.3794** | 0.3074 |
| Median IoU | **0.4304** | 0.0000 |
| IoU >= 0.50 | **32 / 77** | 25 / 77 |
| IoU >= 0.75 | 10 / 77 | **16 / 77** |
| Zero IoU | **25 / 77** | 39 / 77 |
| Empty predicted mask on positive image | **25 / 77** | 39 / 77 |

The main finding is not simply that one method is "better." The deeper result is this:

- **YOLO + SAM2 was more stable and more reliable on this narrow, closed-set aerial task.**
- **SAM3 text prompting was more selective and occasionally excellent, but much less consistent.**
- **When SAM3 fired, it often produced cleaner masks than Pipeline A; when it failed, it often failed completely.**

This is exactly the kind of result that matters for the research question. It suggests that for a domain-specific, single-class remote-sensing problem, **removing the detector is not automatically an upgrade**, even if the new model is concept-aware and text-promptable.

The literature strongly supports this interpretation. Before SAM3, text-prompt segmentation was typically implemented by **text-conditioned detection + SAM** systems such as Grounded SAM. SAM3 changes the landscape by unifying concept recognition and segmentation, but recent follow-up work already shows that **domain-specific adaptation is still needed** for instruction-heavy or remote-sensing scenarios. Our experiment agrees with that direction.

## Research Question

The practical question behind this project is:

> If SAM3 can directly segment objects from a text prompt like `swimming pool`, do we still need a separate detector such as YOLO followed by SAM-style mask refinement?

In a thesis context, the question can be written more precisely as:

> On a single-class aerial segmentation task, is `text prompt + SAM3` a stronger alternative to `supervised detector + SAM2`, or does the detector-then-segment pipeline remain preferable?

This is a meaningful question because it sits at the intersection of two trends:

- the traditional, modular pipeline: **detect first, segment second**
- the new, foundation-model promise: **prompt directly with language**

If SAM3 truly collapses the detection and segmentation stack into one concept-aware model, then a lot of engineering complexity disappears. But if domain shift, small objects, or dense scenes still break direct text prompting, then detector-guided segmentation remains valuable.

## What the Literature Would Lead Us to Expect

### 1. Why detector + SAM became the standard before SAM3

The original SAM was designed as a **promptable segmentation model**, not a native open-vocabulary text-segmentation system. The SAM paper describes a promptable model trained for zero-shot transfer, but the prompts were geometric or mask-like in spirit, not concept-grounded natural language in the modern SAM3 sense.  
Source: <https://arxiv.org/abs/2304.02643>

Once people wanted text-based segmentation, the standard workaround was:

1. use a text-conditioned detector to localize candidate objects
2. pass those boxes into SAM to obtain masks

Grounding DINO explicitly framed open-set detection as bringing language into a detector so it can localize arbitrary concepts or referring expressions.  
Source: <https://arxiv.org/abs/2303.05499>

Grounded SAM then formalized the combined stack: Grounding DINO produces text-conditioned detections and SAM turns them into masks. In other words, **before SAM3, the mainstream text-prompt solution was already a detector + SAM assembly**.  
Source: <https://arxiv.org/abs/2401.14159>

This matters because Pipeline A is not a strange baseline. It is a simplified, closed-set version of a well-established design logic.

### 2. What SAM2 changed

SAM2 extended promptable segmentation to images and videos and improved the underlying segmentation backbone. It was still fundamentally a **promptable visual segmentation** model, not yet a concept-native segmentation system in the SAM3 sense.  
Source: <https://arxiv.org/abs/2408.00714>

So a strong expectation before running this experiment would be:

- `YOLO + SAM2` should be a reasonable closed-set baseline
- but it still depends heavily on detector quality

That expectation matches our setup.

### 3. What SAM3 claims to change

SAM3 is the first SAM-family model that natively targets **Promptable Concept Segmentation (PCS)**. The official SAM3 paper says it can detect, segment, and track based on concept prompts such as short noun phrases or visual exemplars, and that it returns masks for all matching instances.  
Sources:

- <https://arxiv.org/abs/2511.16719>
- <https://github.com/facebookresearch/sam3>
- <https://huggingface.co/docs/transformers/en/model_doc/sam3>

This creates a natural expectation:

- maybe the separate detector is no longer necessary
- maybe text prompting alone can replace the classic two-stage stack

If one only reads the high-level SAM3 framing, the optimistic hypothesis is:

> "For a concept like `swimming pool`, SAM3 should just find and segment all pools directly."

### 4. Why remote sensing should make us cautious

The remote-sensing literature is already more skeptical. Work on applying SAM to aerial or orbital imagery consistently notes domain difficulties:

- extreme scale variation
- dense scenes
- small objects
- lower effective object resolution after resizing
- semantic mismatch between internet-scale pretraining and geospatial scenes

A 2023 remote-sensing SAM study found SAM promising, but explicitly reported limitations on lower spatial resolution data and recommended additional adaptation or fine-tuning.  
Source: <https://arxiv.org/abs/2306.16623>

Later work such as **RS2-SAM2** argues that applying text-conditioned SAM2 to remote sensing requires feature alignment, pseudo-mask prompt generation, and task-specific adaptation because remote-sensing text understanding and prompt generation are themselves challenging.  
Source: <https://arxiv.org/abs/2503.07266>

Recent work such as **SegEarth-OV3** explores training-free SAM3 for remote-sensing open-vocabulary segmentation, but still adds remote-sensing-specific fusion and presence-score filtering to make it practical.  
Source: <https://arxiv.org/abs/2512.08730>

So the literature-based expectation before running our experiment should not have been "SAM3 will obviously crush detector+SAM." The more defensible expectation is:

- **SAM3 should be more elegant**
- **SAM3 may be more general**
- **but on remote sensing, especially for small or dense objects, raw text prompting may still be unstable without adaptation**

That is very close to what we actually observed.

## Experimental Design

## Dataset Choice

We used **iSAID**, filtered to the `swimming pool` class only.

This is a reasonable choice for the research question because iSAID:

- is an **aerial instance segmentation** dataset
- includes `swimming pool` as an official category
- provides pixel-level annotations in COCO-style format

Official category listing confirms `swimming pool` is one of the annotated classes and that annotations are pixel-level.  
Source: <https://captain-whu.github.io/iSAID/dataset.html>

### Local subset statistics

After filtering iSAID to swimming pools only:

| Split | Images | Positive images | Negative images | Pool annotations |
|---|---:|---:|---:|---:|
| Train | 1411 | 259 | 1152 | 2442 |
| Val | 458 | 77 | 381 | 743 |

Positive validation images were also highly uneven:

- mean instances per positive image: `9.65`
- median instances per positive image: `2`
- maximum in a single image: `180`

This is important. The task is not just "find one big pool in a clean scene." Some images contain many small pools dispersed across a large aerial tile.

## Pipelines

### Pipeline A: YOLO + SAM2

- Detector: `YOLOv8n`
- Input size: `1024`
- Epochs: `50`
- Confidence threshold at inference: `0.25`
- Segmenter: `SAM2.1_b`

The logic is:

1. YOLO predicts pool bounding boxes
2. each box is sent to SAM2
3. predicted masks are merged into one binary foreground mask per image

### Pipeline B: SAM3 text prompt

- Model: local `facebook/sam3`
- Prompt: fixed text phrase `swimming pool`
- Threshold: `0.5`
- Mask threshold: `0.5`

The logic is:

1. one text prompt is given directly to SAM3
2. all returned masks are merged into one binary foreground mask per image

## Evaluation Protocol

Ground truth instance masks were merged into a **single binary foreground mask per image**. Predicted masks were merged in the same way. IoU was then computed on the merged masks:

`IoU = intersection / union`

This was implemented as **union-mask IoU**, not instance matching.

That choice is defensible because the prompt is class-level (`swimming pool`) and both pipelines can return multiple instances. But it also means:

- the metric measures **total pool region overlap**
- it does **not** measure instance-wise matching quality

### Important protocol limitation

The main reported IoU summary used `positive_only = true`, so only the `77` validation images with at least one pool were included in the final IoU average.

That means the headline numbers answer:

> "How well do the pipelines segment pools when pools exist?"

They do **not** fully answer:

> "How often does each pipeline hallucinate pools in empty images?"

I therefore include an auxiliary negative-image analysis later in this report.

## Implementation Notes

The relevant local artifacts are:

- Config: `configs/experiment.yaml`
- YOLO training log: `runs/yolo_pool/train/results.csv`
- Final summary: `results/metrics/summary.csv`
- Per-image IoU: `results/metrics/per_image_iou.csv`
- Qualitative overlays: `results/visualizations/`

The local SAM3 pipeline used Hugging Face `Sam3Model` and `Sam3Processor`, text-only prompt input, and `post_process_instance_segmentation` output.  
Source for API behavior: <https://huggingface.co/docs/transformers/en/model_doc/sam3>

Hardware used for the experiment:

- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- VRAM: `8188 MiB`

## YOLO Training Health

## Was YOLO badly trained?

This is the first thing that has to be answered before interpreting the comparison.

If YOLO were obviously under-trained, unstable, or overfit, then Pipeline A would be an unfair baseline. That is **not** what the training log shows.

### Best validation metrics during training

| Metric | Best epoch | Best value |
|---|---:|---:|
| Precision | 11 | 0.9286 |
| Recall | 46 | 0.4280 |
| mAP50 | 46 | 0.6447 |
| mAP50-95 | 48 | 0.3517 |

### Final epoch metrics

| Metric | Value |
|---|---:|
| Precision | 0.8333 |
| Recall | 0.3701 |
| mAP50 | 0.6167 |
| mAP50-95 | 0.3456 |

### Training time

- total wall time: `15,824` seconds
- approximately `4 hours 23 minutes 44 seconds`

### Interpretation

The training health looks acceptable:

- best validation metrics occur late in training, not only at the beginning
- validation losses bottom out around epochs `47-48`
- there is **no clear overfitting signature**

So the problem is not "YOLO training broke."

The more accurate diagnosis is:

- **YOLO learned a usable detector**
- **but it remained recall-limited**

This is a key distinction.

YOLO does not appear to be memorizing and collapsing. Instead, it is simply not recovering enough true pools in this difficult aerial setting. That makes Pipeline A a valid, but imperfect, supervised baseline.

## Main Quantitative Results

## Positive-image IoU results

These are the headline results from `results/metrics/summary.csv`.

| Metric | YOLO + SAM2 | SAM3 text prompt |
|---|---:|---:|
| Mean IoU | **0.3794** | 0.3074 |
| Number of evaluated images | 77 | 77 |

This alone says Pipeline A wins on average. But the per-image distribution is more informative.

## Per-image distribution

| Metric | YOLO + SAM2 | SAM3 text prompt |
|---|---:|---:|
| Mean IoU | **0.3794** | 0.3074 |
| Median IoU | **0.4304** | 0.0000 |
| Std. dev. | 0.3079 | 0.3498 |
| Images with IoU >= 0.25 | **50** | 37 |
| Images with IoU >= 0.50 | **32** | 25 |
| Images with IoU >= 0.75 | 10 | **16** |
| Images with IoU = 0 | **25** | 39 |

### What this means

This table reveals the core behavior difference:

- Pipeline A is **more stable**
- Pipeline B is **more polarized**

SAM3 produced more very high-quality masks (`IoU >= 0.75`) than Pipeline A, but it also produced many more complete misses (`IoU = 0`).

This is why the mean and median favor Pipeline A even though SAM3 has some spectacular successes.

## Head-to-head image comparison

Across the `77` positive validation images:

- Pipeline A wins on `34`
- Pipeline B wins on `22`
- ties on `21`

The mean IoU difference per image was:

- `mean(A - B) = +0.0720`
- bootstrap 95% CI: `[+0.0067, +0.1333]`

This suggests the average advantage of Pipeline A is real in this sample, although the head-to-head win count alone is not overwhelmingly decisive:

- two-sided sign test on non-ties: `p = 0.1409`

Interpretation:

- the comparison is **not** "A crushes B on almost every image"
- instead, **A wins by being more dependable**
- **B loses mostly through more empty-output failures**

## Empty-mask behavior on positive images

On the `77` positive images:

- Pipeline A produced an empty mask on `25`
- Pipeline B produced an empty mask on `39`

Crucially:

- every `IoU = 0` case was an **empty prediction**
- there were no cases where a non-empty mask overlapped zero with the ground truth

So SAM3 did not mostly fail by wildly segmenting the wrong blue shapes. In this experiment it mostly failed by **not committing to any pool mask at all**.

That is an important qualitative distinction.

## Conditional quality when a mask exists

If we look only at positive images where the model produced a non-empty mask:

| Conditional metric | YOLO + SAM2 | SAM3 text prompt |
|---|---:|---:|
| Mean IoU given non-empty prediction | 0.5618 | **0.6228** |
| Median IoU given non-empty prediction | 0.5792 | **0.6512** |

This is one of the most important findings in the entire report.

It means:

- **SAM3 is often very good when it actually decides a pool is present**
- **its main weakness here is recall / coverage, not necessarily mask quality**

In plain language:

- Pipeline A says "yes" more often and therefore covers more pools overall
- Pipeline B says "yes" less often, but when it does, it often segments well

That is exactly the signature of a **conservative but high-precision concept segmenter**.

## Auxiliary analysis on negative images

Because the main IoU summary excludes negative images, I separately measured whether each pipeline produced any mask on the `381` validation images with **no pool at all**.

### Image-level false positive behavior on negative scenes

| Metric | YOLO + SAM2 | SAM3 text prompt |
|---|---:|---:|
| Negative images | 381 | 381 |
| Non-empty mask on negative image | 7 | **1** |
| False-positive scene rate | 1.84% | **0.26%** |

This reinforces the conservativeness story:

- **SAM3 was more cautious**
- **YOLO + SAM2 was more willing to produce a mask**

### Image-level pool-presence classification

Treating "predicted any non-empty mask" as `pool present`:

| Metric | YOLO + SAM2 | SAM3 text prompt |
|---|---:|---:|
| TP | 52 | 38 |
| FP | 7 | **1** |
| FN | **25** | 39 |
| TN | 374 | **380** |
| Precision | 0.8814 | **0.9744** |
| Recall | **0.6753** | 0.4935 |
| Specificity | 0.9816 | **0.9974** |

This provides a very clear behavioral summary:

- **YOLO + SAM2 is the higher-recall system**
- **SAM3 is the higher-precision / higher-specificity system**

This is consistent with the positive-image IoU findings and helps explain why SAM3 can have cleaner conditional masks yet lower average IoU.

## Qualitative Analysis

Representative overlays are available in:

- `results/visualizations/P0104.png`
- `results/visualizations/P0130.png`
- `results/visualizations/P0179.png`
- `results/visualizations/P0086.png`

## Case 1: SAM3 clearly wins

Example: `P0104.png`

- Pipeline A IoU: `0.0000`
- Pipeline B IoU: `0.8892`

Interpretation:

- the pool is visually salient enough that the text concept `swimming pool` is sufficient
- YOLO failed to propose a bounding box, so Pipeline A had no chance to recover
- SAM3 directly recovered a strong mask

This is the strongest argument in favor of direct concept segmentation:

> if the detector misses the object, the whole detector->segmenter stack collapses; a concept-aware segmenter can sometimes bypass that bottleneck.

## Case 2: YOLO + SAM2 clearly wins

Examples:

- `P0179.png`: A `0.6687`, B `0.0000`
- `P1066.png`: A `0.7700`, B `0.0000`

Interpretation:

- these scenes contain small or less visually dominant pools embedded in cluttered aerial context
- the supervised detector is able to localize them
- SAM3 text prompting often stays silent

This is the strongest argument in favor of detector-guided segmentation:

> once the task is narrow and the detector is trained specifically for that class, geometric localization becomes a powerful prior that stabilizes segmentation.

## Case 3: both systems struggle

Example: `P0130.png`

- the image contains many small pools
- both systems underperform
- Pipeline A is less catastrophic, but still not strong

Interpretation:

- dense, tiny-instance remote-sensing scenes remain difficult for both approaches
- this is not merely a YOLO problem or merely a SAM3 problem
- this is a data-resolution and domain-structure problem

## Case 4: both systems succeed

Example: `P0086.png`

- Pipeline A IoU: `0.8845`
- Pipeline B IoU: `0.9001`

Interpretation:

- large, visually clear pools are easy for both systems
- the difference emerges mainly in borderline cases

## Literature Review and Synthesis

## SAM1 established promptable segmentation, not concept-native text segmentation

The original SAM project showed that a promptable segmentation model trained at massive scale can transfer zero-shot to many tasks and image distributions.  
Source: <https://arxiv.org/abs/2304.02643>

However, the conceptual center of SAM1 was:

- flexible prompting
- efficient segmentation
- transferability

not full open-vocabulary concept segmentation from text phrases.

That is why the community quickly gravitated toward **assembled systems**: use a text-grounded detector for localization and use SAM for mask extraction.

## Grounding DINO + Grounded SAM made detector-then-mask the default text solution

Grounding DINO framed open-set detection as language-guided localization of arbitrary objects or referring expressions.  
Source: <https://arxiv.org/abs/2303.05499>

Grounded SAM then made the design pattern explicit: combine Grounding DINO with SAM to detect and segment any region from text input. It also reported strong zero-shot open-vocabulary results.  
Source: <https://arxiv.org/abs/2401.14159>

This means that historically, the answer to "How do I do text prompt segmentation?" was:

> **Use a text detector first, then hand the boxes to SAM.**

That background matters because YOLO + SAM is not conceptually obsolete just because SAM3 exists. It is a closed-set specialization of a pipeline family that has already been validated in open-world research.

## SAM2 strengthened segmentation, but not the concept grounding problem

SAM2 improved promptable segmentation quality and speed, especially for video, through a stronger architecture and data engine.  
Source: <https://arxiv.org/abs/2408.00714>

For our project, this means SAM2 is a sensible choice for the segmentation half of a detector-guided pipeline. But it does not itself solve text grounding.

## SAM3 is the first real attempt to unify concept grounding and segmentation in one model

According to the official SAM3 paper and repository:

- SAM3 is a unified model that can detect, segment, and track objects
- it supports **text prompts** and **visual prompts**
- it aims to segment **all instances** of an open-vocabulary concept
- it is benchmarked on the new SA-Co benchmark with very large concept coverage

Sources:

- <https://arxiv.org/abs/2511.16719>
- <https://github.com/facebookresearch/sam3>
- <https://huggingface.co/docs/transformers/en/model_doc/sam3>

This is the reason the current research question is interesting at all. SAM3 is the first model in this family that makes it plausible to ask:

> "Can I remove the detector and just use language?"

## But the follow-up literature already adds caveats

### Instruction richness is still a problem

SAM3-I argues that base SAM3 is centered on **short noun-phrase prompts**, while richer instructions involving attributes, relations, functions, or reasoning often still require external agents or additional adaptation.  
Source: <https://arxiv.org/abs/2512.04585>

This matters for interpretation:

- if even richer language use still needs adaptation
- then simple concept prompting should not be assumed to be universally robust across domains

### Domain fine-tuning can help, but it is not automatic magic

The official SAM3 repository explicitly supports inference and fine-tuning.  
Source: <https://github.com/facebookresearch/sam3>

Medical follow-up work such as **MedSAM3** reports large gains by fine-tuning SAM3 on medical concept labels and argues that domain-specific promptable concept segmentation is achievable with adaptation.  
Source: <https://arxiv.org/abs/2511.19046>

This is important because it supports the user's intuition:

> "If SAM3 can take text, maybe fine-tuning it would let us mask everything."

But the existence of MedSAM3 does **not** prove that raw SAM3 is enough. It proves the opposite:

> **domain adaptation matters enough that researchers are already building specialized SAM3 variants.**

### Remote sensing remains a hard transfer domain

Remote-sensing SAM papers consistently report that the domain is not plug-and-play:

- lower spatial resolution hurts
- prompt design matters
- domain adaptation helps
- specialized modules are often introduced

Relevant sources:

- SAM in remote sensing: <https://arxiv.org/abs/2306.16623>
- RS2-SAM2: <https://arxiv.org/abs/2503.07266>
- SegEarth-OV3: <https://arxiv.org/abs/2512.08730>

SegEarth-OV3 is especially relevant because it explicitly explores **training-free SAM3** for remote-sensing open-vocabulary segmentation and still adds remote-sensing-specific fusion and presence filtering. That is very close in spirit to our finding that raw text prompting has potential, but needs help to become dependable.

### Even SAM-family mask quality has repeatedly required refinement work

HQ-SAM was proposed because the original SAM, despite massive-scale training, still fell short on intricate structures and mask quality in many cases.  
Source: <https://arxiv.org/abs/2306.01567>

This broader pattern is worth noting:

- large promptable segmentation models are powerful
- but high-quality boundary masks and domain robustness still often require targeted refinement

That pattern fits our result as well.

## What Our Experiment Confirms

Our experiment does **not** show that SAM3 is weak in general.

It shows something narrower and more useful:

### Confirmed finding 1

On a **single-class, closed-set aerial task**, a supervised detector plus SAM-style segmenter can still outperform direct text prompting in **average segmentation reliability**.

This is exactly the scenario where detector-guided pipelines should be strongest:

- the label space is tiny
- the detector can specialize
- geometric prompts reduce ambiguity

### Confirmed finding 2

Direct text-prompt SAM3 can recover objects that the detector misses entirely.

This is the strongest pro-SAM3 result in the experiment. It shows that the concept-based route is not only elegant, but genuinely useful.

### Confirmed finding 3

SAM3's weakness here is mostly **coverage**, not necessarily **mask quality when active**.

That is why SAM3 had:

- lower average IoU
- more zero-IoU cases
- but higher conditional IoU when non-empty

### Confirmed finding 4

The remote-sensing domain shift is real.

The literature says remote sensing requires adaptation, and our result is consistent with that. A general-purpose concept segmentation model did not automatically dominate a narrow supervised baseline.

## What Our Experiment Does Not Prove

It does **not** prove that:

- SAM3 is worse than detector+SAM in all domains
- SAM3 fine-tuning would not help
- YOLO + SAM2 is the best possible baseline

Specifically, our experiment leaves open several possibilities:

1. A fine-tuned SAM3 variant could outperform both pipelines.
2. A stronger detector than `yolov8n` could widen Pipeline A's lead.
3. Higher-resolution or tiled SAM3 inference could reduce its missed detections.
4. Negative prompts, exemplars, or hybrid prompts could improve SAM3 substantially.

## Answer to the Key Practical Question

## "If SAM3 can take text, and we fine-tune it, won't it just mask everything?"

The short answer is:

**No, not automatically.**

A more accurate answer is:

**Fine-tuned SAM3 is a very promising direction, but detector-guided segmentation is still a strong and often better-engineered solution for narrow remote-sensing tasks.**

Why?

### 1. Concept grounding and dense localization are not the same problem

A model may understand the phrase `swimming pool` but still fail to localize every small instance in a dense aerial tile.

### 2. Remote sensing punishes aggressive resizing

The Hugging Face SAM3 documentation warns that custom resolutions may degrade accuracy and notes the model is intended for `1008px` resolution. In aerial imagery, where objects can already be tiny, resizing can erase important evidence.  
Source: <https://huggingface.co/docs/transformers/en/model_doc/sam3>

### 3. Fine-tuning helps, but usually by specializing the model to the domain

That means more data curation, more compute, more training decisions, and more failure modes. It is not free.

### 4. A detector gives a powerful structural prior

For a single known class, a detector narrows the search space:

- where to look
- how many candidates exist
- what regions deserve segmentation attention

That prior is especially valuable when scenes are large and cluttered.

### 5. A hybrid answer is often strongest

The real lesson is not "choose one forever." It is:

- use direct concept segmentation when openness and flexibility matter
- use detector-guided segmentation when stability on a narrow class matters
- consider hybrid designs when you want both

## So what "should" have happened?

Based on the literature, the most defensible expectation before running the experiment would have been:

1. **YOLO + SAM2 should be more stable on a single known class.**
2. **SAM3 should occasionally beat it on examples where the detector misses the object.**
3. **SAM3 should need adaptation before becoming consistently dominant in aerial imagery.**

That is very close to the actual result.

So the experiment did not merely produce numbers. It produced a coherent answer:

> the detector is still doing real work here.

## Why I infer that hybrid or geometry-assisted prompting is often stronger

This section makes explicit which sources support the interpretations used in this report.

### Source-backed interpretation map

#### 1. Why I say "hybrid" means text plus localization cues

This is not an informal guess. The official SAM3 repository itself presents SAM3 as a model that can be prompted not only with text, but also with **visual box prompts**, and it describes the detector as conditioned on **text, geometry, and image exemplars**.  
Source: <https://github.com/facebookresearch/sam3>

That is why I use the phrase **hybrid / geometric support** to refer to setups such as:

- text + box
- text + point
- text + exemplar crop
- detector proposal + SAM3

The claim is grounded in the official model framing, not in analogy alone.

#### 2. Why I say hybrid prompting is often stronger than text-only in remote sensing

The most directly relevant source is the remote-sensing SAM3 study **On the Effectiveness of Textual Prompting with Lightweight Fine-Tuning for SAM3 Remote Sensing Segmentation**. Its abstract states that:

- it compares **textual, geometric, and hybrid prompting strategies**
- **combining semantic and geometric cues yields the highest performance**
- **text-only prompting exhibits the lowest performance**
- textual prompting with light fine-tuning can still be a practical trade-off for **geometrically regular and visually salient targets**

That last point is especially relevant for swimming pools. Pools are often regular and visually salient compared with more amorphous aerial targets, so this paper supports a nuanced interpretation:

- fine-tuned text-only SAM3 could improve a lot on pools
- but the literature still suggests **hybrid semantic+geometric prompting is usually the safer ceiling**

Source: <https://arxiv.org/abs/2512.15564>

#### 3. Why I say domain fine-tuning may help a lot, but not automatically remove the value of geometry

The strongest evidence here comes from **Medical SAM3**. The paper reports that vanilla SAM3 degrades substantially under severe domain shift, and says its apparent competitiveness often relied on **strong geometric priors such as ground-truth-derived bounding boxes**. The paper then motivates **full model adaptation** and reports consistent gains after fine-tuning.  
Source: <https://arxiv.org/abs/2601.10880>

This is exactly why my conclusion is conservative:

- yes, domain fine-tuning can likely improve SAM3 a lot
- no, that does not mean geometry suddenly stops mattering

#### 4. Why I say remote sensing still needs adaptation beyond plain text prompting

The remote-sensing SAM3 paper **SegEarth-OV3** does not fine-tune SAM3, but it still adds task-specific adaptation through:

- mask fusion between SAM3 heads
- presence-score filtering to suppress absent categories

The authors report that these simple additions improve practical behavior in geospatial scenes with dense and small targets. This supports the broader claim that **raw SAM3 is promising, but auxiliary structure still helps in remote sensing**.  
Source: <https://arxiv.org/abs/2512.08730>

#### 5. Why I say detector-guided pipelines are still conceptually justified even in the SAM3 era

This view is supported by two different strands of evidence:

- historically, **Grounding DINO + SAM / Grounded SAM** made text-conditioned detection followed by segmentation the standard open-vocabulary recipe  
  Sources:
  <https://arxiv.org/abs/2303.05499>  
  <https://arxiv.org/abs/2401.14159>
- architecturally, the official SAM3 repository itself does not frame text as the only prompt channel; it explicitly includes geometry and exemplars in the prompting story  
  Source:
  <https://github.com/facebookresearch/sam3>

So when I say "detector + SAM is still a legitimate and often strong design," that is not nostalgia for old pipelines. It is consistent with both the pre-SAM3 literature and SAM3's own official design philosophy.

### Practical synthesis for this project

Putting those sources together, the source-backed version of my interpretation is:

1. **Base text-only SAM3** is a real open-vocabulary segmentation baseline and can sometimes outperform detector-driven systems by bypassing detector misses.
2. **Domain fine-tuning** is likely to improve SAM3 substantially under aerial-domain shift.
3. For remote sensing, the strongest current evidence still points toward **semantic + geometric prompting** as the most reliable form.
4. Therefore, if the goal is absolute performance rather than architectural purity, the strongest future comparison is probably not:

   `YOLO + SAM2` vs `text-only fine-tuned SAM3`

   but rather:

   `YOLO + SAM2` vs `fine-tuned SAM3 + geometric cue`

5. If the goal is to test whether language alone can replace the detector, then **text-only SAM3** remains the correct experimental condition, but it should be interpreted as the stricter and harder setting.

## Limitations

This report should be read with the following caveats.

### 1. YOLO baseline strength

Pipeline A used `yolov8n`, not a larger detector such as `yolov8s` or `yolov8m`. A stronger detector may improve Pipeline A further.

### 2. No SAM3 fine-tuning

The SAM3 evaluation used the base model with a fixed text prompt. This is the correct setup for a zero-shot comparison, but not for judging the ceiling of SAM3 under domain adaptation.

### 3. Positive-only IoU headline

The reported mean IoU excludes negative images. I added negative-scene analysis separately, but the main headline remains a positive-image metric.

### 4. Union-mask IoU rather than instance metrics

This favors image-level region overlap and does not tell us whether individual pool instances were matched one by one.

### 5. One prompt only

The prompt was intentionally fixed as `swimming pool`. That keeps the experiment clean, but it means prompt engineering was not explored.

### 6. Single dataset, single class

This is the right scope for the thesis experiment, but it limits generalization.

## Recommended Next Experiments

If the goal is to turn this into a strong thesis chapter, the next experiments should be:

### 1. Fine-tune or adapt SAM3 for aerial imagery

This is the most important missing experiment if you want to answer the stronger question:

> "Can adapted SAM3 replace detector-guided segmentation in remote sensing?"

### 2. Strengthen the detector baseline

Train `yolov8s` or `yolov8m` and rerun Pipeline A. This tests whether the current result is limited by the detector capacity or by the overall design itself.

### 3. Add full-scene evaluation beyond positive-only IoU

Report:

- image-level precision / recall for `pool present`
- false positive rate on negative images
- instance-level metrics if feasible

### 4. Try tiled or higher-resolution SAM3 inference

This is especially relevant in aerial imagery because SAM3 is designed around `1008px` usage, and small pools may disappear after resizing.

### 5. Try hybrid prompting

Examples:

- text + negative box
- text + exemplar crop
- text + detector proposals

This would test whether the future is really "SAM3 alone" or rather "SAM3 with lightweight localization assistance."

## Final Conclusion

The cleanest conclusion from this project is:

> On the `iSAID swimming pool` validation subset, **YOLO + SAM2** was the better **average** segmentation system, while **SAM3 text prompting** was the more **conservative and occasionally sharper** system.

This means:

- **Detector-guided segmentation is still highly competitive, and in this setup preferable, for narrow aerial tasks.**
- **Direct text-prompt segmentation with SAM3 is promising but not yet a drop-in replacement here.**
- **The right future question is not "SAM3 or detector+SAM forever?" but "what amount of domain adaptation or hybridization lets SAM3 match detector-guided stability without losing its open-vocabulary advantages?"**

That is a strong thesis result because it is neither trivial nor vague. It shows:

- what the literature suggested
- what this experiment tested
- what was confirmed
- what remains unresolved

## Local Artifacts

- `configs/experiment.yaml`
- `runs/yolo_pool/train/results.csv`
- `results/metrics/summary.csv`
- `results/metrics/per_image_iou.csv`
- `results/visualizations/`

## Sources

Primary and official sources used in this report:

1. Segment Anything (SAM): <https://arxiv.org/abs/2304.02643>
2. Grounding DINO: <https://arxiv.org/abs/2303.05499>
3. Grounded SAM: <https://arxiv.org/abs/2401.14159>
4. SAM 2: <https://arxiv.org/abs/2408.00714>
5. SAM 3: <https://arxiv.org/abs/2511.16719>
6. Official SAM3 repository: <https://github.com/facebookresearch/sam3>
7. Hugging Face SAM3 docs: <https://huggingface.co/docs/transformers/en/model_doc/sam3>
8. iSAID dataset site: <https://captain-whu.github.io/iSAID/dataset.html>
9. SAM for remote sensing: <https://arxiv.org/abs/2306.16623>
10. RS2-SAM2: <https://arxiv.org/abs/2503.07266>
11. SegEarth-OV3: <https://arxiv.org/abs/2512.08730>
12. HQ-SAM: <https://arxiv.org/abs/2306.01567>
13. SAM3-I: <https://arxiv.org/abs/2512.04585>
14. MedSAM3: <https://arxiv.org/abs/2511.19046>

## Current-Status Note

As of **2026-03-29**, the official SAM3 repository also lists a **2026-03-27** update announcing `SAM 3.1 Object Multiplex`. This experiment used the base `facebook/sam3` checkpoint, not SAM 3.1, so the result should be interpreted as a comparison against **SAM3**, not the latest SAM 3.1 release.  
Source: <https://github.com/facebookresearch/sam3>
