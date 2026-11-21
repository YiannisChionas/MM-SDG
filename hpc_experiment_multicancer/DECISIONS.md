# Multi-cancer paper-scale reproduction — decision log

Living document. Every decision made about how this experiment is run
gets added here, in order, with the reasoning, so we never have to
re-derive "why did we choose X" later. Update this file whenever a new
decision is made or an old one is revised (note the revision, don't just
overwrite silently).

## 1. Cancer types: COAD, SCCRCC, EPM, LIHB, IDC (not the paper's LUAD/KIRP/CESC/COAD/GBM)

The paper's own 5 (LUAD, KIRP, CESC, COAD, GBM) don't hold up against the
real HEST-1k catalog: KIRP has 0 Visium samples, CESC has 1, GBM has 3-4.
Replaced with the 5 oncotree codes with the most Visium samples instead:
COAD (50), SCCRCC (24), EPM (14), LIHB (10), IDC (10).

**Why**: can't reproduce what doesn't exist in the data; picking by
actual sample availability is the only defensible alternative.

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

**Status**: not yet run for this 5-cancer-type pool -- needs its own
gene-selection pass (common genes across these 5 types specifically, not
reusing the CCRCC-only 14,166-gene panel).

## Open / not yet decided

- VAE architecture settings for this 5-cancer-type pool (may differ from
  `hpc_experiment_vae_paper`'s CCRCC-only settings once trained on a
  different, larger, multi-type gene panel/dataset).
- Patient-level vs WSI-level train/val/test split -- LIHB's `patient`
  metadata field was found empty for all its rows during the earlier
  oncotree survey, so patient-level splitting may not be possible for
  that type specifically; needs a fallback (WSI-level) for LIHB.
- Diffusion model settings (architecture, conditioning mechanism, cascade
  stages) for this multi-cancer setup.
