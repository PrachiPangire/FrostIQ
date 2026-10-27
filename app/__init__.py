"""FrostIQ Flask website: landing page, telemetry console and JSON API."""

from flask import Flask

from frostiq.config import DEFAULT_THRESHOLD_C, RAW_DATA_DIR
from frostiq.model import DEFAULT_MODEL_PATH
from frostiq.simulate import READINGS_FILE

from app.services import JourneyStore, ModelProvider

# Verified from this repository's Git remote (origin).
REPO_URL = "https://github.com/PrachiPangire/FrostIQ"


def create_app(readings_path=None, model_path=None, threshold_c=DEFAULT_THRESHOLD_C) -> Flask:
    """Build the app. Paths default to the project's data/ and models/ folders."""
    app = Flask(__name__)
    app.config.update(
        READINGS_PATH=readings_path or RAW_DATA_DIR / READINGS_FILE,
        MODEL_PATH=model_path or DEFAULT_MODEL_PATH,
        DEFAULT_THRESHOLD_C=float(threshold_c),
        REPO_URL=REPO_URL,
        # Re-read templates when they change, so edits show up without restarting.
        # (Python code changes still need a restart.)
        TEMPLATES_AUTO_RELOAD=True,
    )
    app.json.sort_keys = False  # keep API fields in a readable order
    app.extensions["frostiq_store"] = JourneyStore(app.config["READINGS_PATH"])
    app.extensions["frostiq_models"] = ModelProvider(app.config["MODEL_PATH"])

    from app.routes import bp
    app.register_blueprint(bp)
    return app
