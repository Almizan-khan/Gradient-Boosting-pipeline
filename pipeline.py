import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def prepare_features(frame, categorical):
    result = frame.copy()
    for column in categorical:
        result[column] = result[column].fillna('__MISSING__').astype(str)
    return result

def train_cv(frame, target, identifier='id', test=None, folds=5, seed=42, iterations=3000, output=None):
    from catboost import CatBoostRegressor
    if target not in frame or frame[target].isna().any():
        raise ValueError('A non-missing target column is required')
    if not 2 <= folds <= len(frame):
        raise ValueError('folds must be between 2 and the number of rows')
    feature_names = [c for c in frame if c not in {identifier, target}]
    if not feature_names:
        raise ValueError('At least one feature is required')
    x = frame[feature_names]
    categorical = x.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    x = prepare_features(x, categorical)
    y = frame[target].astype(float)
    if not np.isfinite(y).all():
        raise ValueError('Targets must be finite')
    if test is not None:
        missing = set(feature_names)-set(test.columns)
        if missing:
            raise ValueError('Missing test features: '+', '.join(sorted(missing)))
        test = prepare_features(test[feature_names], categorical)
    oof = np.zeros(len(x))
    predictions = np.zeros(len(test)) if test is not None else None
    rows, importances = [], []
    destination = Path(output) if output else None
    if destination:
        destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for fold, (train_index, valid_index) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(x)):
        model = CatBoostRegressor(iterations=iterations, learning_rate=0.03, depth=8, loss_function='RMSE',
                                  cat_features=categorical, random_seed=seed, early_stopping_rounds=200,
                                  verbose=False, thread_count=2, allow_writing_files=False)
        model.fit(x.iloc[train_index], y.iloc[train_index], eval_set=(x.iloc[valid_index], y.iloc[valid_index]), use_best_model=True)
        oof[valid_index] = model.predict(x.iloc[valid_index])
        if predictions is not None:
            predictions += model.predict(test)/folds
        rows.append({'fold': fold+1, 'rmse': float(np.sqrt(mean_squared_error(y.iloc[valid_index], oof[valid_index]))),
                     'best_iteration': model.get_best_iteration(), 'validation_rows': len(valid_index)})
        importances.append(model.get_feature_importance())
        if destination:
            model.save_model(str(destination/f'fold_{fold+1}.cbm'))
    report = {'rmse': float(np.sqrt(mean_squared_error(y, oof))), 'folds': rows,
              'feature_importance': dict(zip(feature_names, np.mean(importances, axis=0).tolist())),
              'features': feature_names, 'categorical_features': categorical,
              'config': {'folds': folds, 'seed': seed, 'iterations': iterations, 'learning_rate': 0.03, 'depth': 8},
              'seconds': time.perf_counter()-started, 'environment': {'python': platform.python_version(), 'platform': platform.platform()}}
    if destination:
        (destination/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
        pd.DataFrame({'target': y.to_numpy(), 'prediction': oof}).to_csv(destination/'oof_predictions.csv', index=False)
    return report, predictions

def predict_saved(frame, directory):
    from catboost import CatBoostRegressor
    directory = Path(directory)
    report = json.loads((directory/'metrics.json').read_text())
    features = prepare_features(frame[report['features']], report['categorical_features'])
    files = sorted(directory.glob('fold_*.cbm'))
    if len(files) != report['config']['folds']:
        raise ValueError('Checkpoint directory does not contain the expected fold models')
    predictions = []
    for path in files:
        model = CatBoostRegressor()
        model.load_model(str(path))
        predictions.append(model.predict(features))
    return np.mean(predictions, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test')
    parser.add_argument('--target', default='exam_score')
    parser.add_argument('--id', default='id')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='outputs/catboost')
    args = parser.parse_args()
    frame = pd.read_csv(args.train)
    test = pd.read_csv(args.test) if args.test else None
    report, predictions = train_cv(frame, args.target, args.id, test, args.folds, args.seed, args.iterations, args.output)
    if predictions is not None:
        if args.id not in test:
            raise ValueError('Test file must contain the ID column')
        pd.DataFrame({args.id: test[args.id], args.target: predictions}).to_csv(Path(args.output)/'submission.csv', index=False)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
