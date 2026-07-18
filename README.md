# Prosodiff

[![PyPI](https://img.shields.io/pypi/v/prosodiff?label=PyPI)](https://pypi.org/project/prosodiff/)
[![Python](https://img.shields.io/pypi/pyversions/prosodiff)](https://pypi.org/project/prosodiff/)
[![Tests](https://github.com/asera-1/prosodiff/actions/workflows/tests.yml/badge.svg)](https://github.com/asera-1/prosodiff/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/asera-1/prosodiff/blob/main/LICENSE)

**Same words. Different delivery—made measurable.**

Prosodiff is a training-free CLI and loopback-only browser interface for comparing
two to four recordings of the same speaker saying the same text. It exports a 4:5
PNG, SVG, or PDF report plus versioned JSON containing per-take measures, explicit
pairwise deltas, contours, and reliability diagnostics.

> Prosodiff describes recorded acoustic differences. It does not infer emotion,
> linguistic boundaries, listener response, or an overall expressivity score.

Built as a side tool for retrieval-augmented expressive TTS research (ProsodyRAG) at Universität Osnabrück.

![Prosodiff social preview](https://raw.githubusercontent.com/asera-1/prosodiff/main/docs/prosodiff-social-preview.png)

## Quick start

Run the complete synthetic demonstration without cloning or a persistent install:

```bash
uvx prosodiff demo --output prosodiff-card.png
```

This creates `prosodiff-card.png` and `prosodiff-card.json`; no audio is required.
The first run may take one to two minutes while the scientific Python stack builds
local caches.

For a permanent installation:

```bash
pip install prosodiff
```

Prosodiff requires Python 3.10 or newer.

## Example report

![Prosodiff synthetic demonstration](https://raw.githubusercontent.com/asera-1/prosodiff/main/docs/prosodiff-card.png)

The example is generated from deterministic synthetic signals. No recordings or
research datasets are committed. Its machine-readable output is available as
[`docs/prosodiff-card.json`](https://github.com/asera-1/prosodiff/blob/main/docs/prosodiff-card.json).

## What Prosodiff measures

| Dimension | Reported as | Interpretation boundary |
| --- | --- | --- |
| Duration | Seconds and pairwise percentage deltas | Not speaking rate; text and trimming are not verified |
| Pitch | Median F0 and IQR span in semitones | Unavailable when reliable pYIN coverage is insufficient |
| Recorded energy | Active RMS level in dBFS and within-take dynamics | Sensitive to microphone, gain, distance, and AGC |
| Pauses | Internal gap count, duration, and share | Energy-defined gaps, not linguistic pause labels |

Every unordered pair is reported explicitly as **B − A**. The report also surfaces
pitch coverage, clipping, recorded-level, pitch-boundary, duration-mismatch, and
pause-sensitivity notices instead of silently hiding unreliable measurements.

## Compare recordings

Record the same sentence back-to-back, then run:

```bash
prosodiff compare calm.wav encouraging.wav \
  --label Calm \
  --label Encouraging \
  --text "Heute schaffen wir das gemeinsam." \
  --output prosodiff-card.png
```

Pass two, three, or four WAV files and repeat `--label` once per file. The first
take is the delta reference. By default, Prosodiff writes the JSON report beside
the figure with the same filename stem. Run `prosodiff compare --help` for pitch
bounds, confidence-display settings, and a custom JSON output path.

## Record live locally

Start the browser interface after installation:

```bash
prosodiff ui
```

Live recording is the primary path: record, replay, redo, and label two to four
takes before generating the same figure and JSON as the CLI. Existing WAV upload
remains available as a secondary option.

![Prosodiff local browser interface](https://raw.githubusercontent.com/asera-1/prosodiff/main/docs/prosodiff-ui.png)

The interface:

- binds only to `127.0.0.1` and has no authentication;
- makes no telemetry, CDN, or external API requests;
- stores recordings in a temporary run directory and deletes them after analysis;
- keeps generated results only for the local server session;
- is intended for local use, not remote deployment.

If the browser does not open automatically, use the URL printed in the terminal.
For a fixed port:

```bash
prosodiff ui --no-open-browser --port 7860
```

## Recording protocol

Between-take recorded-level comparisons are interpretable only when every take
uses:

1. the same speaker and exact text;
2. the same device, room, gain, microphone direction, and distance;
3. one back-to-back recording session;
4. disabled auto-gain, noise suppression, echo cancellation, and voice enhancement
   where possible;
5. unclipped, unnormalized WAV files.

Prosodiff cannot verify these assumptions. It records available capture provenance
in JSON and marks microphone-sensitive measures in the figure.

## Method

Prosodiff resamples analysis copies to 22.05 kHz. It does not peak-normalize,
denoise, pre-emphasize, repair pitch, or time-warp audio.

- **Pitch:** `librosa.pyin`, 50–600 Hz by default. F0 follows pYIN's
  Viterbi-decoded voiced flag inside the utterance and outside detected pauses.
  Summaries require at least 20 decoded frames and 25% eligible-frame coverage.
  Median F0 uses a take-balanced shared semitone reference; pitch span is the IQR.
- **Recorded energy:** time-domain RMS in dBFS. Absolute active-frame level is
  retained for pairwise deltas; displayed contours are relative to the pooled take
  median. This is recorded level, not calibrated SPL, loudness, or vocal effort.
- **Energy-defined pauses:** an adaptive RMS threshold, removal of sub-50 ms
  energy islands, and filling of sub-150 ms gaps. Internal gaps use a nominal
  150 ms cutoff with 11.6 ms hop quantization. A ±3 dB sensitivity check generates
  a notice when results change materially.
- **Time:** contours use raw seconds from detected utterance onset. They are not
  phonetic alignment. Normalized coordinates in JSON are a coarse linear-time
  view, not phone correspondence.

## Interpretation limits

- pYIN probabilities are engineering diagnostics, not calibrated certainty.
- Recording-chain changes can dominate recorded-energy differences.
- Browser and operating-system audio processing may remain active even when the
  recorder requests that it be disabled.
- Same-speaker and same-text assumptions are supplied by the user and unverified.
- Short-utterance acoustic deltas do not establish listener perception,
  population effects, or cognitive outcomes.

## JSON contract

The machine-readable report uses schema
`prosodiff.explicit-delivery-attribute-delta`, version `0.2.0`. It contains:

- fixed analysis parameters and recording assumptions;
- input metadata, utterance spans, acoustic attributes, and coded notices;
- raw-time and normalized-time contours, candidate F0, voiced flags, masks, and
  probability diagnostics;
- every unordered pair in input order, identified as `b_minus_a`;
- pair-level reliability fields and potential confounds.

Unavailable values are encoded as JSON `null`. There is no aggregate expressivity
or quality score. Prosodiff is currently alpha software; schema changes will remain
versioned and will be documented in releases.

## Development

```bash
git clone https://github.com/asera-1/prosodiff.git
cd prosodiff
uv sync --extra dev
uv run pytest
uv run prosodiff demo --output docs/prosodiff-card.png
```

Tests use deterministic synthetic fixtures created inside temporary directories.

## Roadmap

1. Add optional transcript-aware phone alignment while retaining raw-time views.
2. Add batch manifests for ProsodyRAG evaluation and Flowent tutor-voice QA.
3. Validate acoustic deltas against preregistered listener judgments.

## Citation, author, and license

If Prosodiff supports your work, cite the software metadata in
[`CITATION.cff`](https://github.com/asera-1/prosodiff/blob/main/CITATION.cff).

Created by **Abdalla Sera** ([asera-1](https://github.com/asera-1)). Prosodiff's
authored source is released under the
[MIT License](https://github.com/asera-1/prosodiff/blob/main/LICENSE).

Dependencies retain their own licenses. The locked librosa audio stack may install
dynamically used LGPL components such as `libsndfile` and `python-soxr`; they are
not vendored or redistributed by this repository.
