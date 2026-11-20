# RNA-Conditioned Cascaded Super-Resolution (Stage 2) — 224×224

The RNA-conditioning analogue of [hpc_experiment_sr](../hpc_experiment_sr):
same cascaded idea (low-res stage-1 → upsample → super-resolution stage-2
at native 224×224), but conditioned on the frozen beta-VAE's gene
expression embedding ([hpc_experiment_rna](../hpc_experiment_rna)'s
mechanism) instead of a discrete cancer-type label.

## Why this exists

Two results needed reconciling:

1. At 64×64: RNA-conditioning beat class-conditioning on CCRCC (FID 167.5
   vs 184.4, and on the *harder* held-out-WSI protocol).
2. Cascading the class-conditioned model to native 224×224
   (`hpc_experiment_sr`) made FID **worse**, not better (137.3 → 155.05 pooled) —
   evaluating at higher fidelity exposed more of a too-small, error-
   compounding cascade's flaws rather than rewarding the better evaluation
   protocol.

Open question: does RNA-conditioning's advantage hold up, strengthen, or
also break down once run through the same cascade? This experiment
answers that.

## Architecture

- `Stage1RnaConditionedUnet`: exact copy of `hpc_experiment_rna`'s
  `RnaConditionedUnet` (64×64), duplicated here (not imported across
  folders) so this experiment stays self-contained and can load that
  experiment's already-trained `checkpoint.pt` directly.
- `RnaSRConditionedUnet`: the SR-stage analogue of `hpc_experiment_sr`'s
  `SRConditionedUnet`, but conditioned on the frozen VAE's embedding
  (broadcast) instead of a class `nn.Embedding`. `in_channels = 3 (noisy) +
  3 (low-res cond) + 16 (VAE latent) = 22`. Same deliberately-small channel
  width (64 base) as `hpc_experiment_sr`, same reasoning (time budget at
  224×224).
- Training data: real patches degraded (downsample→upsample) exactly like
  `hpc_experiment_sr`, but now paired with the matching spot's real gene
  expression via `hpc_experiment_rna`'s barcode-matching index. CCRCC
  train-split WSIs only (official HEST-bench fold 0).

## Conditioning augmentation (added after the first result)

Same fix as `hpc_experiment_sr` (see that README for the full
rationale): the regressor-alignment result showed the cascade's
expression-fidelity degrading much more than its FID did (+24-28% →
+49% MAE), consistent with stage 2 never seeing anything but a perfectly
clean degraded real image during training. `HestPatchExprPairDataset`
now adds random-strength Gaussian noise to the low-res conditioning
during training (`--cond-aug-max-std`, default 0.2, matching Ho et al./
Imagen's noise conditioning augmentation). Retraining with this change is
still pending -- the numbers in this README predate it.

## Time budget

CCRCC train-split is ~55,808 patches (measured from `hpc_experiment_rna`'s
own run) vs `hpc_experiment_sr`'s pooled 236,495 -- about 23.6% the size.
Scaling that experiment's **measured** yoda throughput (51min/epoch, after
it turned out 2.8x faster than obiwan's MIG slices): **~12min/epoch
estimated here**. 10 epochs ≈ 2h, comfortable within a single 4h job --
unlike `hpc_experiment_sr`, this one likely doesn't need
`submit_train.sh`-style chaining, but it's provided as a fallback (and
`--persist-dir` per-epoch sync is kept regardless, now standard practice
after a lost epoch in that experiment).

## Cascaded sampling

`sample_rna_sr.py` loads **three** checkpoints:
1. `hpc_experiment_rna`'s `vae_checkpoint.pt` (frozen beta-VAE)
2. `hpc_experiment_rna`'s `checkpoint.pt` (`Stage1RnaConditionedUnet`, 64×64)
3. This experiment's `checkpoint.pt` (`RnaSRConditionedUnet`, 224×224)

Same held-out WSI protocol as `hpc_experiment_rna`'s `sample_rna.py`:
2000 samples (matched to that experiment's own `--max-samples`, for a
fair before/after-cascade comparison), real patches saved at **native**
224×224 (no downsampling — the whole point).

## Usage

```bash
ln -s ~/git/MM-SDG/hpc_experiment_rna_sr ~/hest-rna-sr
mkdir -p ~/experiments/hest_rna_sr

sbatch -J hest_rna_sr_train -o ~/experiments/hest_rna_sr/train.out -e ~/experiments/hest_rna_sr/train.err \
    slurm/job_train.sh --epochs 10
# If that times out, fall back to: ./slurm/submit_train.sh

sbatch -J hest_rna_sr_sample -o ~/experiments/hest_rna_sr/sample.out -e ~/experiments/hest_rna_sr/sample.err \
    slurm/job_sample.sh

sbatch -J hest_rna_sr_fid -o ~/experiments/hest_rna_sr/fid.out -e ~/experiments/hest_rna_sr/fid.err \
    slurm/job_fid.sh
```

Results land in `/opt/storage/users/${USER}/experiments/hest-rna-sr/`.

## Comparison this enables

Both rows for class-conditioning use its **CCRCC-specific** FID (from
`hpc_experiment`/`hpc_experiment_sr`'s per-class breakdown), not the
10-cancer-type pooled number -- RNA-conditioning is CCRCC-only throughout
(gene panels/splits are per-cancer-type in HEST-bench), so comparing
against the pooled class-conditioned number would mix scopes.

| | FID (CCRCC only) |
|---|---|
| Class-conditioned, 64×64 | 184.4 |
| Class-conditioned, 224×224 cascade | 282.86 (much worse, not just slightly) |
| RNA-conditioned, 64×64 (held-out WSIs) | 167.5 |
| **RNA-conditioned, 224×224 cascade (this), held-out WSIs** | **267.25** |

## Result

RNA-conditioning's advantage **holds up** through the cascade: 267.25 <
282.86, so it stays ahead of class-conditioning at native resolution too,
same as it was at 64×64 (167.5 vs 184.4).

That said, both conditioning strategies degrade sharply once cascaded to
224×224 (class: 184.4→282.86, +53%; RNA: 167.5→267.25, +60%) — comparably
dramatic drops. This points to the shared bottleneck being the cascade
itself / the deliberately small (64-base-channel) stage-2 SR model, not
the conditioning signal: conditioning determines the relative ranking
(RNA ahead in both regimes) but doesn't offset the capacity loss from
undersizing the SR stage. This matches the RNA-CDM paper's own stage-2
being considerably wider (128 base channels vs our 64) — a strong
candidate explanation, and the motivation for training a paper-scale
model (see the main repo README for the larger-partition request).
