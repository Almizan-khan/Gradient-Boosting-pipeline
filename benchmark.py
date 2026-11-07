import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def run(include_catboost=True):
    dataset = load_diabetes(as_frame=True)
    x, y = dataset.data, dataset.target
    rows = []
    for seed in [17, 42, 73]:
        split = KFold(5, shuffle=True, random_state=seed)
        for name, model in [('mean', DummyRegressor()), ('ridge', make_pipeline(StandardScaler(), Ridge(alpha=10.0)))]:
            started = time.perf_counter()
            predictions = cross_val_predict(model, x, y, cv=split)
            rows.append({'model': name, 'seed': seed, 'rmse': float(np.sqrt(mean_squared_error(y, predictions))), 'seconds': time.perf_counter()-started})
        if include_catboost:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(iterations=100, learning_rate=0.03, depth=8, loss_function='RMSE', random_seed=seed,
                                      verbose=False, thread_count=2, allow_writing_files=False)
            started = time.perf_counter()
            predictions = cross_val_predict(model, x, y, cv=split)
            rows.append({'model': 'catboost', 'seed': seed, 'rmse': float(np.sqrt(mean_squared_error(y, predictions))), 'seconds': time.perf_counter()-started})
    return {'dataset': 'scikit-learn diabetes regression dataset', 'rows': len(x), 'features': list(x.columns),
            'scope': 'small public reference dataset; not the Kaggle student-score competition',
            'protocol': '5-fold out-of-fold RMSE, three seeds; fixed parameters; CatBoost always trains 100 iterations without evaluation-fold early stopping',
            'caveat': 'Small single-dataset comparison, not a general model ranking. The competition pipeline retains validation-based checkpoint selection separately.',
            'results': rows, 'environment': {'python': platform.python_version(), 'platform': platform.platform()}}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='reports/benchmark.json')
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    result = run(not args.baseline_only)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
