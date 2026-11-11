# causal_clean_v1

Low-cost first-pass cleaning for the bidirectional true-streaming S2TT
trajectory data. It does not modify 'clean_exact_v1' or any original label,
alignment, or packed artifact.

## Files

- streaming/data/clean_causal_trajectories.py
  joins mapping + LLM chunk labels + source alignment, filters one direction,
  applies conservative boundary relabeling, and writes unpacked train/valid.
- streaming/data/audit_causal_trajectories.py
  validates causal metadata, direction, turn structure, contiguous timing and
  the 640-ms grid; it also writes human-readable samples.
- Existing streaming/data/pack_streaming_s2tt.py
  performs exact-runtime-length packing.
- Existing streaming/data/balance_mix_packed.py
  downsamples the larger direction and mixes packed bins exactly 50:50.
- streaming/scripts/prepare_causal_clean_v1.sh
  is the stage orchestrator.

## Direction-specific filters

Both directions are processed independently before mixing.

Common rules:

1. LLM label has monotonicity == 0 ("strictly monotonic" in the original
   labeling prompt), no failed review, and no bad reviewed boundary.
2. Unreviewed labels require confidence >= 90.
3. Source forced alignment score >= 95.
4. Timestamp tokens reconstruct the source transcript exactly.
5. At most 5% zero-duration timestamp tokens, no negative/non-monotonic time,
   no token longer than 2 s, and aligned speech spans at least 35% of the clip.
6. At least two distinct independent ASR transcripts agree with the source.
7. Target-prefix progress cannot lead source-prefix progress beyond the
   direction threshold.
8. Existing structural/language/rate checks still pass.
9. After relabeling and structural merging, at least two interleave turns
   remain.

Defaults that differ by direction:

| direction | ASR error | max target lead |
|---|---:|---:|
| En->Zh | WER <= 0.25 | 0.18 |
| Zh->En | CER <= 0.12 | 0.15 |

Each non-final WRITE is delayed by 640 ms by default. The last boundary remains
the true end of the source clip. Boundaries landing on the same grid are merged,
never reordered.

This v1 uses the existing LLM monotonicity judgment and a prefix-progress
causality proxy, not a newly downloaded SimAlign model. A true cross-lingual
word-alignment/NIR scorer should be a v2 after the inexpensive A/B experiment
confirms the data hypothesis. Target semantic adequacy is partially inherited
from the upstream COMET-filtered pool; v1 removes structural, ASR, alignment,
fluency-shape, and trajectory defects but does not claim to replace a
model-based translation-QE pass.

## Smoke

Run the complete small-data pipeline:

~~~bash
cd /path/to/megatron
SMOKE_LIMIT=256 bash streaming/scripts/prepare_causal_clean_v1.sh smoke
~~~

It performs cleaning, causal audit, structural audit, packing, packing
conservation, exact-length audit and exact 50:50 packed-bin mixing. A unique
streaming/artifacts/causal_clean_v1_smoke_* directory is created.

## Full pipeline

Run each resumable stage into a new root:

~~~bash
cd /path/to/megatron

STREAMING_CAUSAL_ROOT=streaming/artifacts/causal_clean_v1 \
bash streaming/scripts/prepare_causal_clean_v1.sh clean

STREAMING_CAUSAL_ROOT=streaming/artifacts/causal_clean_v1 \
bash streaming/scripts/prepare_causal_clean_v1.sh audit

STREAMING_CAUSAL_ROOT=streaming/artifacts/causal_clean_v1 \
bash streaming/scripts/prepare_causal_clean_v1.sh pack

STREAMING_CAUSAL_ROOT=streaming/artifacts/causal_clean_v1 \
bash streaming/scripts/prepare_causal_clean_v1.sh mix
~~~

Or run all four stages once:

~~~bash
STREAMING_CAUSAL_ROOT=streaming/artifacts/causal_clean_v1 \
bash streaming/scripts/prepare_causal_clean_v1.sh all
~~~

Final training lists:

- streaming/artifacts/causal_clean_v1/bidirectional/packed/train/data.list
- streaming/artifacts/causal_clean_v1/bidirectional/packed/valid/data.list

The mix uses equal numbers of packed bins, which balances training token budget
more directly than balancing raw utterance count.

## Threshold overrides

All major thresholds are environment-overridable:

~~~bash
BOUNDARY_DELAY_MS=640
MIN_LABEL_CONFIDENCE=90
MIN_ALIGNMENT_SCORE=95
MIN_ASR_VOTES=2
CLEAN_LIMIT=0
PACK_WORKERS=16
~~~

Direction-specific ASR and target-lead thresholds can be passed directly to
clean_causal_trajectories.py; defaults are recorded in every
cleaning_summary.json, and record-level values are stored under
streaming.causal_clean.
