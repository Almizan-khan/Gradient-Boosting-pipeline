# Tabular regression · CatBoost

A cross-validation pipeline for mixed numeric and categorical features, with out-of-fold predictions, saved fold models, and baseline comparisons.

[![checks](https://github.com/Almizan-khan/Gradient-Boosting-pipeline/actions/workflows/checks.yml/badge.svg)](https://github.com/Almizan-khan/Gradient-Boosting-pipeline/actions/workflows/checks.yml)


## What this demonstrates

- Five-fold model evaluation and an ensemble of fold predictions.
- Consistent categorical/missing-value handling and explicit ID/target exclusion.
- Mean feature importance across folds, reloadable models, and schema checks.

## Quick start

Use Python 3.11. Obtain the competition files through your own Kaggle account and accept its dataset terms there.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python GRADIENTBOOST.py --train data/train.csv --test data/test.csv --target exam_score --id id --output outputs/student-scores
```

On Windows, activate with `.venv\Scripts\activate`. Input paths are configurable; no `/kaggle/input` mount is required. `pipeline.py` provides the same CLI and reusable `train_cv` / `predict_saved` functions.

## Outputs

| Artifact | Purpose |
|---|---|
| `metrics.json` | Overall/fold RMSE, configuration, features and timing |
| `oof_predictions.csv` | One held-out prediction per training row |
| `fold_*.cbm` | Reloadable CatBoost models |
| `submission.csv` | ID-aligned average predictions for the supplied test file |

Models and downloaded data remain outside source control. The original five-fold approach is retained; numerical results must be generated from the actual supplied dataset.

## Reproducible reference benchmark

```bash
python benchmark.py --output reports/benchmark.json
python -m unittest discover -s tests -v
```

This uses scikit-learn's bundled diabetes regression dataset, so it needs no Kaggle credentials. It compares a mean predictor, scaled Ridge regression and CatBoost with five-fold out-of-fold RMSE over three seeds. It is a small reference experiment, not a clinical tool or a result on the student-score competition. CI publishes the generated report as an artifact.

The historical student-score CV RMSE of 8.75719 is retained in [the original README](docs/original-readme.md), but is not represented as independently reproduced. The maintained benchmark reports its evaluation protocol and limitations alongside the numbers.

## Measured reference results

Five-fold OOF RMSE over seeds 17, 42 and 73; lower is better. Parameters were fixed, and the CatBoost reference model uses exactly 100 iterations without validation-fold early stopping.

| Model | Mean RMSE | Sample SD across seeds |
|---|---:|---:|
| mean | 77.230 | 0.178 |
| ridge | 54.850 | 0.265 |
| catboost | 57.499 | 0.459 |

[Raw results and provenance](reports/diabetes-fixed-budget.json) · [Executed run](https://github.com/Almizan-khan/Gradient-Boosting-pipeline/actions/runs/42401858794)

Ridge outperformed this CatBoost configuration on this dataset. That is a useful baseline finding, not a reason to omit the simpler model. Three correlated re-splits are not a confidence interval or a universal model ranking. The competition training pipeline separately retains its original validation-based checkpoint selection.

## Design and limitations

Feature importance is averaged across folds and describes this fitted model, not causal influence. Random K-fold validation assumes rows are suitably independent; use group or time-aware splits when the data requires them. For model-selection studies, separate tuning from final evaluation.

See [validation](VALIDATION.md) and the [original experiment](notebooks/original_experiment.ipynb).

## References

- [CatBoost documentation](https://catboost.ai/docs/)
- [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
