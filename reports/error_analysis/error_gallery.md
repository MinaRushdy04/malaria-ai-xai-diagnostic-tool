# Error Analysis Gallery

Generated: 2026-05-11 15:08 UTC

Threshold: `0.285`

This gallery exports the highest-risk false negatives and false positives from the test split.
False negatives are sorted by highest Parasitized score below threshold; false positives are
sorted by highest Parasitized score above threshold.

## Counts

- False negatives exported: 8
- False positives exported: 8

## Files

- [Gallery index](error_gallery_index.csv)
- Images folder: `images/`

## Why This Matters

Healthcare AI projects should inspect failure modes directly. Aggregate metrics are necessary,
but reviewing false positives and false negatives helps reveal whether errors are related to
image quality, morphology, threshold placement, or model attention patterns.
