# Task 4 registration experiments

`config.py` is the single place to select an active dataset, pair(s), feature,
registration method, and output location. It contains ready-to-use blocks for
Bunny, 3DMatch, ScanNet, and PAVIN.

The execution boundary is:

```
datasets/ -> preprocessing/feature_pipeline.py -> main.py registration dispatch -> evaluation/
```

Dataset adapters resolve paths/formats and return clouds. Registration methods
do not know the dataset name. Add a dataset by implementing `DatasetHandler`
and registering it in `datasets/handlers.py`; add a feature by returning the
standard `prepare_registration_inputs` dictionary; add a registration method by
adding one dispatch branch in `main.py` (or, preferably, a runner function).

Run one pair with `python3 task4/main.py`, the configured pairs with
`python3 task4/batch.py`, or a product sweep with
`python3 task4/run_combinations.py`. The sweep calls the batch API directly and
never edits `config.py`.

Batch runs are intentionally quiet on success. Runtime limits for feature
RANSAC and DGR's fallback RANSAC are explicit global settings in `config.py`;
they prevent CPU runs on large datasets from appearing to hang.

For DGR + Robust ICP, selected keypoints/descriptors produce the coarse input
and/or initial seed. Since DGR has no descriptor input argument, the seed is
applied to DGR's input and its residual transform is composed back into the
original frame. Robust ICP then receives that DGR transform as its initial pose
and refines the dense downsampled clouds.
