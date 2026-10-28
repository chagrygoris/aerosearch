# Gemma 4 native video vs frame sampling — Kaggle T4, 2026-10-28

Run: `saudabubin/aerosearch-gemma4-video-bench-v2` · model `google/gemma-4-E4B-it`, nf4 4-bit,
fp16 compute · `transformers` 5.21.0 · Tesla T4 (15360 MiB, sm_75).

Both arms at `max_soft_tokens=140` **per frame**, so the comparison is budget-matched per frame
but not in total: native sampled 16 frames, the baseline 6.

| clip | mode | sec | input tokens |
|---|---|---|---|
| blazes | native (`num_frames=16`) | 16.3 | 2101 |
| blazes | 6 sampled frames | 8.6 | 770 |
| silent_motion | native (`num_frames=16`) | 11.9 | 2261 |
| silent_motion | 6 sampled frames | 9.1 | 830 |

## Captions

**blazes — native:** "…a promotional sequence for HBO GO, featuring clips from Game of Thrones,
such as scenes with Daenerys Targaryen and dragons, **interspersed with** shots of using a
Chromecast device. The video **concludes with** branding for HBO GO, Chromecast, and Google…"

**blazes — 6 frames:** "…promotional material for **"Game of Thrones on HBO GO,"** showcasing
scenes of fire and characters… accompanied by on-screen text like **"FOR BIGGER BLAZES."**"

**silent_motion — native:** "…vertical stripes of various colors… The circular elements contain
numbered or lettered markers that **change throughout the clip**, suggesting a visual sequence or
**progression**."

**silent_motion — 6 frames:** "…concentric arcs of bright colors… suggesting a digital or glitch
art aesthetic."

## What this shows

1. **Native video does deliver temporal understanding.** It is the only arm that describes
   ordering and change — "interspersed with", "concludes with", "change throughout the clip".
   Frame sampling produces a static scene description with no sense of sequence. This is the
   capability the rebuild wanted for silent, motion-only clips.
2. **But native missed on-screen text the frame arm read.** The 6-frame arm quoted
   "FOR BIGGER BLAZES" and "Game of Thrones on HBO GO" verbatim; native named the franchise but
   quoted no on-screen text. At 140 soft tokens/frame, more frames did not buy legible text.
   Research guidance is 70-140 for captioning and 1120 for OCR, and that split shows here.
3. **Native costs ~2x wall-clock and ~2.7x input tokens** for those 16 frames.

For aerosearch this matters because the corpus is Russian-language memes where on-screen text is
often the whole joke. Temporal description and OCR pull in opposite directions at a fixed budget.

## Caveat

Two clips only — one public promo, one synthetic ffmpeg test pattern, neither Russian and neither
from `src-videos/`. Directional, not conclusive.

## Next experiment

Vary `max_soft_tokens` (140 vs 560 vs 1120) across both arms on real `src-videos` clips, shipped
as a Kaggle Dataset, and score Cyrillic OCR fidelity alongside temporal description.

## Two API facts worth keeping

- **`fps` is unusable without also clearing `num_frames`.** `Gemma4VideoProcessor` defaults
  `num_frames=32`, and the two are mutually exclusive, so passing `fps` alone still raises
  `ValueError: num_frames and fps are mutually exclusive`. Both runs here failed that arm.
- **Processor kwargs must go in a `processor_kwargs` dict.** Passed as `**kwargs` to
  `apply_chat_template`, transformers 5.21 warns and ignores them, so `max_soft_tokens` silently
  has no effect.

## Cost model for the batch worker

Model load 144-152 s; captioning 9-16 s/clip. `gemma-4-E4B-it` at nf4 occupies **9.23 GB**, fitting
a T4 with headroom. No `HF_TOKEN` needed (Apache 2.0). A 200-clip batch is roughly 35-60 min of
GPU, dominated by per-clip inference rather than load.
