import tempfile
import unittest
import numpy as np
import pandas as pd
from pipeline import prepare_features, train_cv, predict_saved

class PipelineTests(unittest.TestCase):
    def test_missing_categories_are_consistent(self):
        frame = pd.DataFrame({'kind': ['x', None], 'number': [1, 2]})
        result = prepare_features(frame, ['kind'])
        self.assertEqual(result.kind.tolist(), ['x', '__MISSING__'])
        self.assertTrue(frame.kind.isna().any())

    def test_training_and_saved_inference_agree(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame({'id': range(40), 'x': rng.normal(size=40), 'kind': ['a', 'b']*20})
        frame['target'] = frame.x*3
        test = frame.iloc[:5].drop(columns='target')
        with tempfile.TemporaryDirectory() as folder:
            result, predictions = train_cv(frame, 'target', test=test, folds=2, iterations=8, output=folder)
            np.testing.assert_allclose(predictions, predict_saved(test, folder))
            self.assertNotIn('id', result['features'])
            self.assertNotIn('target', result['features'])
            self.assertEqual(sum(f['validation_rows'] for f in result['folds']), 40)

    def test_missing_target_rejected(self):
        with self.assertRaises(ValueError):
            train_cv(pd.DataFrame({'x': [1, 2]}), 'target')

    def test_missing_inference_feature_rejected(self):
        with self.assertRaises(ValueError):
            train_cv(pd.DataFrame({'x': [1, 2], 'target': [1, 2]}), 'target', folds=2, test=pd.DataFrame({'wrong': [1]}))
