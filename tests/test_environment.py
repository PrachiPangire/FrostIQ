"""Smoke tests: confirm the environment and core libraries work together."""

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def test_python_version():
    assert sys.version_info >= (3, 14)


def test_frostiq_package_importable():
    import frostiq

    assert frostiq.__version__ == "0.1.0"


def test_pandas_numpy_roundtrip():
    df = pd.DataFrame({"temp_c": np.linspace(2.0, 8.0, 7)})
    assert df["temp_c"].max() == 8.0


def test_sklearn_regressor_fits():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = 2 * X.ravel()
    model = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y)
    assert model.predict([[10.0]]).shape == (1,)


def test_flask_app_responds():
    from flask import Flask

    app = Flask(__name__)
    app.add_url_rule("/", view_func=lambda: "ok")
    assert app.test_client().get("/").data == b"ok"
