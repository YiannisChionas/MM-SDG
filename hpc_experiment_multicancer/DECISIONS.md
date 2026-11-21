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

**Cleanup**: done -- the 10 mouse LIHB `.h5`/`.h5ad` files were deleted
from `/opt/storage/datasets/HEST/` (`patches_multicancer/` and `st/`).

### 1c. Second bug found the same way: one targeted-panel WSI (TENX28) collapsed the gene panel to 1,064

After switching to PRAD and re-running gene selection, the cross-type
intersection came back as only **1,064 genes** -- still far too small
(for comparison, intersecting just one representative WSI per type gave
~17,311). Checked `n_vars` for all 47 selected WSIs individually: **one
COAD WSI, `TENX28`, has only 1,142 genes** (a targeted/custom probe
panel, not full transcriptome) vs. 17,943-36,601 for every other WSI.
A single such outlier anywhere in the pool is enough to collapse a
join='inner' intersection, regardless of how good the other 46 WSIs are.

**Fix**: added `--min-genes` (default 15,000) to
`compute_extractable_patches.py`, excluding any WSI below that
threshold from the candidate pool entirely, before WSI selection ever
sees it. TENX28 dropped, COAD's candidate pool went 50 -> 49, `select_wsis.py`
picked a replacement automatically (closest-to-median among the
remaining 49). Re-ran the full chain (recompute extractable patches ->
reselect WSIs -> extract the newly-selected WSIs, clean up the
no-longer-selected ones -> reselect genes).

**Result**: common-gene intersection across all 5 types, all 47 WSIs:
**14,368 genes** (close to the CCRCC-only pool's 14,166 from
`hpc_experiment_vae_paper`, as expected now that the outlier is gone).

**Lesson for later**: always sanity-check `n_vars` isn't a wild outlier
across a to-be-pooled set before trusting an intersection -- a single
bad WSI can silently wreck it, and the result (a very small but nonzero
panel) doesn't look obviously broken unless you check.

## 2. Up to 10 WSIs per cancer type, selected to balance physical tissue coverage

For types with more candidates than the target (COAD, SCCRCC, EPM), the
N WSIs whose patch count is closest to the overall cross-type median are
kept (not just the largest N), so each type's total data volume ends up
comparable rather than arbitrary. PRAD only has 7 human Visium
candidates total, so all 7 are used.

**Why**: avoids one type dominating total data volume just because more
raw samples happened to be available for it.

**Final selection (47 WSIs, after the LIHB and TENX28 fixes above)**:

| Type | WSIs | Final patches (post-extraction, post-4K-cap) |
|---|---|---|
| COAD | 10 | 34,013 |
| SCCRCC | 10 | 31,564 |
| EPM | 10 | 27,515 |
| PRAD | 7 | 26,733 |
| IDC | 10 | 35,041 |
| **Total** | **47** | **154,866** |

Range 26,733-35,041 (1.31x) -- well balanced.

## 3. Patch size: 128µm physical size → resized to 256×256px (matches the paper exactly)

256px is the paper's tile size; 0.5µm/px is the paper's target
resolution; 256 × 0.5 = 128µm per side. Each WSI's own
`pixel_size_um_estimated` determines the RAW pixel crop window needed to
cover 128µm (e.g. ~464px for COAD at 0.276µm/px, ~144-280px for
PRAD/SCCRCC at ~0.61-0.46µm/px); that raw crop is then resized (down or
up) to the fixed 256×256 tensor.

**Why 128µm, not something larger**: an alternative (~236µm) was
considered to avoid ANY upsampling (even for the coarsest-resolution
type's worst case), but rejected -- it would degrade effective
resolution for every WSI, including the already-good COAD/IDC ones,
just to fix 1-2 types. Not justifiable to make everything worse for
everyone to fix a localized issue in 1-2 types.

**Known consequence, accepted as a documented limitation**: PRAD and
EPM will frequently be upsampled (~1.5-1.8x, their native resolution is
coarser than 0.5µm/px) -- those patches are genuinely lower-detail than
COAD/SCCRCC/IDC's, reflecting real differences in source-cohort scanner
resolution, not an artifact of our pipeline. (Original measurement was
against LIHB, since replaced by PRAD -- PRAD's resolution profile is
similar: ~0.34-0.62µm/px, mostly needing upsampling like EPM did.)

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

Paper: *"a maximum of randomly chosen 4000 tiles per slide"*. On the
final 47-WSI selection (COAD/SCCRCC/EPM/PRAD/IDC), several WSIs exceed
4,000 (IDC and PRAD WSIs especially, given PRAD's coarser resolution
means each WSI has fewer *unique* pixels but comparable spot density;
see section 2's final table for post-cap per-type totals: 154,866 total
across all 5 types, range 26,733-35,041, 1.31x).

The cap is applied during extraction itself (`extract_patches.py`), not
as a separate post-processing step.

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

**Decision**: leave the post-cap per-type totals (26,733-35,041, 1.31x
range) as-is for training -- matches the paper's own practice, avoids
discarding real data for an artificial equal-count target.

## 7. Remaining per-type imbalance handled via weighted DataLoader sampling, not further truncation

To avoid the model being subtly biased toward over-represented types
during training (IDC/COAD vs PRAD/EPM), use weighted/balanced sampling in
the DataLoader (e.g. inverse-frequency weights per cancer type) so each
training batch sees roughly equal representation across types, without
physically discarding any patches.

**Status**: RESOLVED. `data_utils_multicancer.make_weighted_sampler`
builds a `WeightedRandomSampler` with inverse cancer-type-frequency
weights, used for the *fit* split's DataLoader in
`train_vae_multicancer.py`. Smoke-tested locally: correctly rebalances
a synthetic 10:1 class imbalance down to ~1:1 in the drawn samples.

## 8. Gene panel: common genes across the 5 cancer types + QC filter only, no HVG cutoff

Confirmed from the paper's own Methods: *"we selected those genes the
different cancer types have in common"* then basic filtering, ending at
17,655 (a consequence of their process, not a chosen target). Matches
exactly what we already decided for the CCRCC-only paper-scale VAE
(`hpc_experiment_vae_paper`): intersect genes across all pooled WSIs,
apply `sc.pp.filter_genes` only (no highly-variable-gene cutoff, since
that was measured to discard ~78% of expression variance on our data).

**Status**: RESOLVED. Ran `select_genes_multicancer.py` on the final
47-WSI pool (after the LIHB and TENX28 fixes above): common genes across
all 5 types = 15,979, after `filter_genes(min_cells=1%)` = **14,368
genes** -- comparable to the CCRCC-only pool's 14,166
(`hpc_experiment_vae_paper`), confirming the two bugs above (mouse LIHB,
targeted-panel TENX28) were the actual cause of the earlier near-zero
intersections, not some fundamental incompatibility between platforms.

## 9. Train/val/test split -- RESOLVED (patient-level where possible, WSI-level fallback)

`data_utils_multicancer.split_train_test_wsis` checks, per cancer type,
whether the `patient` field is populated for every one of its **selected**
WSIs; if so it splits by unique patient (keeping all of a patient's WSIs
on one side), otherwise it falls back to a WSI-level split for that type.

**Checked directly on our final 47 selected WSIs** (not the broader
candidate pool, which is a different, noisier picture -- e.g. COAD's
full candidate pool had some patient info across *some* rows, but the
specific 10 we selected for balance turned out to all have the field
empty):

| Type | Patient info in our final selected WSIs |
|---|---|
| COAD | empty -- WSI-level fallback |
| SCCRCC | 10 unique patients (1.00 WSI/patient) -- patient-level split |
| EPM | 9 unique patients (1.11 WSI/patient) -- patient-level split |
| PRAD | empty -- WSI-level fallback |
| IDC | empty -- WSI-level fallback |

So **3 of 5 types** (COAD, PRAD, IDC) fall back to WSI-level splitting in
practice. The split function itself needed no change for this -- it
already checks per-type, not hardcoded to any specific type. This is a
real, accepted limitation for those 3 types: a WSI-level split can't
rule out two sections from the same patient landing on opposite sides,
but there's no way to do better without patient IDs that HEST doesn't
provide for these rows.

Implemented in `train_vae_multicancer.py` (used for both the outer
train/test split and the inner train/val split for early stopping).
Smoke-tested locally (no patient leakage where patient-level split
applies; valid splits from the WSI-level fallback). Re-verified against
the final PRAD-based selection (same pattern as the LIHB-based one:
2 of 5 types have usable patient info, 3 don't).

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
