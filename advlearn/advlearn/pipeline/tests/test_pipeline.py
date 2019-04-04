"""Test the pipeline module"""

import numpy as np
import pytest
from advlearn.pipeline import Pipeline
from advlearn.defenses import DummyDefense
from advlearn.classifier import DummyClassifier

X = np.array(
    [
        [-0.12840393, 0.66446571],
        [1.32319756, -0.13181616],
        [0.04296502, -0.37981873],
        [0.83631853, 0.18569783],
        [1.02956816, 0.36061601],
        [1.12202806, 0.33811558],
        [-0.53171468, -0.53735182],
        [1.3381556, 0.35956356],
        [-0.35946678, 0.72510189],
        [1.32326943, 0.28393874],
        [2.94290565, -0.13986434],
        [0.28294738, -1.00125525],
        [0.34218094, -0.58781961],
        [-0.88864036, -0.33782387],
        [-1.10146139, 0.91782682],
        [-0.7969716, -0.50493969],
        [0.73489726, 0.43915195],
        [0.2096964, -0.61814058],
        [-0.28479268, 0.70459548],
        [1.84864913, 0.14729596],
        [1.59068979, -0.96622933],
        [0.73418199, -0.02222847],
        [0.50307437, 0.498805],
        [0.84929742, 0.41042894],
        [0.62649535, 0.46600596],
        [0.79270821, -0.41386668],
        [1.16606871, -0.25641059],
        [1.57356906, 0.30390519],
        [1.0304995, -0.16955962],
        [1.67314371, 0.19231498],
        [0.98382284, 0.37184502],
        [0.48921682, -1.38504507],
        [-0.46226554, -0.50481004],
        [-0.03918551, -0.68540745],
        [0.24991051, -1.00864997],
        [0.80541964, -0.34465185],
        [0.1732627, -1.61323172],
        [0.69804044, 0.44810796],
        [-0.5506368, -0.42072426],
        [-0.34474418, 0.21969797],
    ]
)
Y = np.array(
    [
        1,
        2,
        2,
        2,
        1,
        1,
        0,
        2,
        1,
        1,
        1,
        2,
        2,
        0,
        1,
        2,
        1,
        2,
        1,
        1,
        2,
        2,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        0,
        2,
        2,
        2,
        2,
        1,
        2,
        0,
    ]
)
X_TEST_1 = np.array(
    [
        [-0.12840393, 0.66446571],
        [1.32319756, -0.13181616],
        [0.04296502, -0.37981873],
        [0.83631853, 0.18569783],
    ]
)
X_TEST_2 = np.array(
    [
        [-0.12840393, 0.66446571],
        [1.32319756, -0.13181616],
        [0.04296502, -0.37981873],
        [0.83631853, 0.18569783],
        [1.02956816, 0.36061601],
        [1.12202806, 0.33811558],
        [-0.53171468, -0.53735182],
        [1.3381556, 0.35956356],
        [-0.35946678, 0.72510189],
        [1.32326943, 0.28393874],
    ]
)


class TestPipeline(object):
    """Test the pipeline module
    """

    @pytest.fixture()
    def pipeline(self):
        """Setup a test pipeline"""
        steps = [
            ("DummyDefense", DummyDefense()),
            ("DummyClassifier", DummyClassifier()),
        ]
        return Pipeline(steps)

    def test_fit(self, pipeline):
        """Test that the pipeline can fit data"""
        pipeline.fit(X, Y)

    def test_partial_fit(self, pipeline):
        """Test that the pipeline can partially fit data"""
        pipeline.fit(X, Y)
        pipeline.partial_fit(X, Y)

    def test_predict(self, pipeline):
        """Test that the pipeline can predict on data"""
        pipeline.fit(X, Y)
        y_out_fit = pipeline.predict(X_TEST_1)
        assert isinstance(y_out_fit, np.ndarray)
        assert y_out_fit.ndim == 1
        pipeline.partial_fit(X, Y)
        y_out_partial_fit = pipeline.predict(X_TEST_2)
        assert isinstance(y_out_partial_fit, np.ndarray)
        assert y_out_partial_fit.ndim == 1
