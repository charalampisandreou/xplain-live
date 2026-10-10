# Model artifacts

The xPass model file is **not in git**: it is 134.8 MB, over GitHub's 100 MB file limit.

Put both files from the same xPass-Project training run in `models/xpass/`:

- `xpass_production_<timestamp>.joblib`: calibrated XGBoost pass-completion model (ignored by git)
- `feature_list.json`: the exact feature columns, in order, the model expects (tracked)

The feature builder rejects a `feature_list.json` whose columns or order differ from
`xplain.features.FEATURE_ORDER`. Load the model with the pinned versions in `requirements.txt`.

TODO: download location for the .joblib (GitHub release asset or Azure Blob).
