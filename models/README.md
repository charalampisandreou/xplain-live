# Model artifacts

The xPass model is trained in this repo, on generated matches only (`scripts/make_dataset.py`,
then `python -m xplain.train`; see the main README). The scorer loads, by default,
`models/xpass_v2/` (`config.XPASS_MODEL_PATH`):

- `xpass_v2.joblib`: CalibratedClassifierCV (isotonic, 5 match-grouped folds) over XGBClassifier
  (ignored by git; rebuild it with the commands above)
- `feature_list.json`: the exact feature columns, in order, the model expects
- `metrics.json`: training settings (chosen depth and tree count, validation log-loss)

Set `XPLAIN_MODEL_PATH` (and optionally `XPLAIN_FEATURE_LIST`) to load another model file.
The feature builder rejects a `feature_list.json` whose columns or order differ from
`xplain.features.FEATURE_ORDER`. Load the model with the pinned versions in `requirements.txt`.
