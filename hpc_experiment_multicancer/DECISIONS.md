# Multi-cancer paper-scale reproduction — decision log

Living document. Every decision made about how this experiment is run
gets added here, in order, with the reasoning, so we never have to
re-derive "why did we choose X" later. Update this file whenever a new
decision is made or an old one is revised (note the revision, don't just
overwrite silently).

## 1. Cancer types: COAD, SCCRCC, EPM, PRAD, IDC (not the paper's LUAD/KIRP/CESC/COAD/GBM)

The paper's own 5 (LUAD, KIRP, CESC, COAD, GBM) don't hold up against the
real HEST-1k catalog: KIRP has 0 Visium samples, CESC has 1, GBM has 3-4.
Replaced with the 5 oncotree codes with the most Visium samples instead:
originally COAD (50), SCCRCC (24), EPM (14), LIHB (10), IDC (10).

**Why**: can't reproduce what doesn't exist in the data; picking by
actual sample availability is the only defensible alternative.

### 1b. LIHB discovered to be 100% mouse -- replaced with PRAD

After extraction, gene-panel selection across the pooled 50 WSIs came
back with **zero common genes**. Diagnosis: LIHB's gene names
(`Xkr4`, `Gm1992`, `Gm37381`, ...) are mouse (MGI) nomenclature, not
human -- checked the metadata's `species` column directly: **all 10
selected LIHB Visium WSIs are `Mus musculus`, 0 are `Homo sapiens`**.
None of our selection scripts had ever filtered on species, only on
`st_technology`/`oncotree_code` -- the oncotree code alone doesn't imply
species, and HEST pools both human and mouse samples under the same
disease codes.

Re-surveyed Visium samples with `species == 'Homo sapiens'` required:
COAD (50), SCCRCC (24), EPM (14), IDC (10), **PRAD (7)**, BLCA (6). PRAD
chosen over BLCA: PRAD's estimated total patches after the 4,000/slide
cap (~26,733) falls in the same range as our other 4 types (27k-39k);
BLCA's (~6,217) would be 4-6x smaller than every other type, far worse
imbalance than anything the weighted sampler is meant to fix.

**Final 5**: COAD, SCCRCC, EPM, **PRAD**, IDC.

**Fix applied going forward**: `species == 'Homo sapiens'` is now a
required filter in every selection script in `hest_dataset_prep/`
(`compute_extractable_patches.py`, `download_adata_only.py`,
`estimate_adata_size.py`), not just `st_technology`/`oncotree_code`.
Never pick a cancer type without checking this column first.

**Cleanup**: the 10 mouse LIHB `.h5` patch files (~17GB, extracted
before this was caught) need deleting from
`/opt/storage/datasets/HEST/patches_multicancer/` -- shared cluster
storage, no reason to keep data we can't use.

**NOTE on everything below**: sections 2-9 still describe the numbers
from the LIHB-based selection (WSI picks, patch counts, 4K-cap totals,
upsampling %, patient-split results). These need recomputing for PRAD
-- re-running `select_wsis.py` -> `extract_patches.py` ->
`select_genes_multicancer.py` -- before the numbers below are trusted.
Will be updated with real PRAD numbers once that's done, not guessed.

## 2. 10 WSIs per cancer type, selected to balance physical tissue coverage

For types with >10 Visium samples (COAD, SCCRCC, EPM), the 10 WSIs whose
patch count is closest to the overall cross-type median are kept (not
just the largest 10), so each type's total data volume ends up
comparable rather than arbitrary.

**Why**: avoids one type dominating total data volume just because more
raw samples happened to be available for it.

## 3. Patch size: 128µm physical size → resized to 256×256px (matches the paper exactly)

256px is the paper's tile size; 0.5µm/px is the paper's target
resolution; 256 × 0.5 = 128µm per side. Each WSI's own
`pixel_size_um_estimated` determines the RAW pixel crop window needed to
cover 128µm (e.g. ~464px for COAD at 0.276µm/px, ~144px for LIHB at
~0.9µm/px); that raw crop is then resized (down or up) to the fixed
256×256 tensor.

**Why 128µm, not something larger**: an alternative (~236µm) was
considered to avoid ANY upsampling (even for LIHB's worst case), but
rejected -- it would degrade effective resolution for every WSI,
including the already-good COAD/SCCRCC/IDC ones, just to fix 1-2 types.
Not justifiable to make everything worse for everyone to fix a
localized issue in 1-2 types.

**Known consequence, accepted as a documented limitation**: LIHB (and
part of EPM) will always be upsampled (~1.6-1.8x) since their native
resolution is coarser than 0.5µm/px -- those patches are genuinely
lower-detail than COAD/SCCRCC/IDC's, reflecting real differences in
source-cohort scanner resolution, not an artifact of our pipeline.
Measured: COAD/SCCRCC/IDC mostly downsampled (safe); LIHB 100% of WSIs
need upsampling; EPM ~86% of WSIs need upsampling.

## 4. Patch-to-expression correspondence is unaffected by crop size

Each patch is centered on a Visium spot's pixel coordinates
(`pxl_row_in_fullres`/`pxl_col_in_fullres` in that WSI's own `.h5ad`),
which is also how that spot's expression row is identified (barcode
match). The crop window size changes how much surrounding tissue is
shown, not which spot (and therefore which expression vector) the patch
belongs to. Confirmed the real Visium spot diameter is only ~55µm
(median, from HEST metadata) -- far smaller than our 128µm patch window,
so even HEST's own pre-made 224px patches already show more context
than the spot itself; this isn't a new issue we introduced.

## 5. Max 4,000 tiles per WSI (random subsample when exceeded) -- matches the paper exactly

Paper: *"a maximum of randomly chosen 4000 tiles per slide"*. Checked
our 50 selected WSIs against this: **13/50 exceed 4,000** (concentrated
in LIHB 5/10, IDC 6/10, SCCRCC 2/10; COAD and EPM are all under).
Applying the cap changes per-type totals:

| Type | Before cap | After cap |
|---|---|---|
| COAD | 32,936 | 32,936 |
| EPM | 27,515 | 27,515 |
| IDC | 38,812 | 35,041 |
| LIHB | 37,249 | 32,823 |
| SCCRCC | 30,382 | 29,513 |
| **Total** | 166,894 | **157,828** |

Side effect: this also improves cross-type balance on its own (range
1.41x -> 1.27x), without us doing anything extra -- it's just a
consequence of applying the paper's own rule.

**How the 4,000 are chosen (for the 13 WSIs that exceed it)**: plain
uniform random subsampling without replacement, over the WSI's already
extractable spots (in-tissue + crop window fits), fixed seed=0 applied
independently per WSI (`extract_patches.py`'s
`np.random.RandomState(seed).choice(..., replace=False)`). No
stratification by spatial location, tissue morphology, or expression
value -- matches the paper's own wording exactly ("maximum of randomly
chosen 4000 tiles per slide"), and is deterministic/reproducible given
the same seed.

## 6. No deliberate further truncation to force equal counts per cancer type

Checked the paper: it does **not** balance training-set tile counts
across cancer types either -- the only cap is the per-slide 4,000 limit
above; whatever totals result per type from slide counts is what they
train on. The "10,000 tiles per cancer type" figure in the paper is for
a *downstream evaluation* (HoverNet comparison), not training data.

**Decision**: leave the post-cap per-type totals (27,515-35,041, 1.27x
range) as-is for training -- matches the paper's own practice, avoids
discarding real data for an artificial equal-count target.

## 7. Remaining per-type imbalance handled via weighted DataLoader sampling, not further truncation

To avoid the model being subtly biased toward over-represented types
during training (IDC/LIHB vs EPM), use weighted/balanced sampling in the
DataLoader (e.g. inverse-frequency weights per cancer type) so each
training batch sees roughly equal representation across types, without
physically discarding any patches.

**Status**: not yet implemented -- to be added to the training script
once it exists.

## 8. Gene panel: common genes across the 5 cancer types + QC filter only, no HVG cutoff

Confirmed from the paper's own Methods: *"we selected those genes the
different cancer types have in common"* then basic filtering, ending at
17,655 (a consequence of their process, not a chosen target). Matches
exactly what we already decided for the CCRCC-only paper-scale VAE
(`hpc_experiment_vae_paper`): intersect genes across all pooled WSIs,
apply `sc.pp.filter_genes` only (no highly-variable-gene cutoff, since
that was measured to discard ~78% of expression variance on our data).

**Status**: script written (`select_genes_multicancer.py`), not run yet --
pending a free job slot.

## 9. Train/val/test split -- RESOLVED (patient-level where possible, WSI-level fallback)

`data_utils_multicancer.split_train_test_wsis` checks, per cancer type,
whether the `patient` field is populated for every one of its **selected**
WSIs; if so it splits by unique patient (keeping all of a patient's WSIs
on one side), otherwise it falls back to a WSI-level split for that type.

**Checked directly on our final 50 selected WSIs** (not the broader
108-candidate pool, which is a different, noisier picture -- e.g. COAD's
full 50-sample pool had ~5 unique patients across *some* rows, but the
specific 10 we selected for balance turned out to all have the field
empty):

| Type | Patient info in our 10 selected WSIs |
|---|---|
| COAD | empty -- WSI-level fallback |
| SCCRCC | 10 unique patients (1.00 WSI/patient) -- patient-level split |
| EPM | 9 unique patients (1.11 WSI/patient) -- patient-level split |
| LIHB | empty -- WSI-level fallback |
| IDC | empty -- WSI-level fallback |

So **3 of 5 types** (COAD, LIHB, IDC) fall back to WSI-level splitting in
practice, not just LIHB as originally thought. The split function itself
needed no change for this -- it already checks per-type, not hardcoded
to LIHB. This is a real, accepted limitation for those 3 types: a
WSI-level split can't rule out two sections from the same patient landing
on opposite sides, but there's no way to do better without patient IDs
that HEST doesn't provide for these rows.

Implemented in `train_vae_multicancer.py` (used for both the outer
train/test split and the inner train/val split for early stopping).
Smoke-tested locally (no patient leakage where patient-level split
applies; valid splits from the WSI-level fallback).

(Earlier version of this note incorrectly said only LIHB needed the
fallback -- corrected after checking the real selected-WSI data instead
of assuming from the broader candidate-pool survey.)

## Open / not yet decided

- **VAE architecture settings for this 5-cancer-type pool**: the
  architecture/hyperparameters themselves are already decided (same
  paper-matched recipe as `hpc_experiment_vae_paper`: BatchNorm+LeakyReLU,
  beta=0.005, 6000/4000/200, lr=3e-3+warmup/cosine, early stopping --
  `BetaVAEPaper` in `data_utils_multicancer.py` is a direct copy). What's
  NOT yet validated is whether these settings (tuned/reported for the
  paper's own data, not ours) actually behave well at this scale/pool --
  posterior collapse, convergence speed, overfitting -- which can only be
  checked by actually running it.
- **Batch size / partition feasibility on anakin/luke**: never actually
  benchmarked. Needs a real run to confirm.
- Diffusion model settings (architecture, conditioning mechanism, cascade
  stages) for this multi-cancer setup -- not started at all yet (VAE
  first).
