"""Page routes and JSON API for the FrostIQ website."""

from flask import Blueprint, current_app, jsonify, render_template, request

from frostiq.features import DEFAULT_HORIZON_MIN
from frostiq.model import MIN_HISTORY_MIN

from app.services import (
    RECENT_READINGS,
    WARNING_MARGIN_C,
    DataUnavailable,
    InvalidRequest,
    journey_payload,
    model_card,
    parse_threshold,
)

bp = Blueprint("frostiq", __name__)


def _store():
    return current_app.extensions["frostiq_store"]


def _models():
    return current_app.extensions["frostiq_models"]


def _default_threshold():
    return current_app.config["DEFAULT_THRESHOLD_C"]


def _api_error(message, status):
    return jsonify({"error": message}), status


# --- Pages -----------------------------------------------------------------


@bp.get("/")
def index():
    try:
        dataset, data_error = _store().dataset_info(), None
    except DataUnavailable as err:
        dataset, data_error = None, str(err)
    _, model_error = _models().get()
    card = model_card(_models())
    return render_template("index.html", dataset=dataset, data_error=data_error,
                           card=card, model_error=model_error, threshold_c=_default_threshold(),
                           horizon_min=card["horizon_min"] if card else DEFAULT_HORIZON_MIN,
                           repo_url=current_app.config["REPO_URL"])


@bp.get("/dashboard")
def dashboard():
    try:
        journey_ids, data_error = _store().journey_ids(), None
    except DataUnavailable as err:
        journey_ids, data_error = [], str(err)
    requested = request.args.get("journey", "")
    selected = requested if requested in journey_ids else (journey_ids[0] if journey_ids else "")
    _, model_error = _models().get()
    card = model_card(_models())
    return render_template("dashboard.html", data_error=data_error, model_error=model_error,
                           horizon_min=card["horizon_min"] if card else DEFAULT_HORIZON_MIN,
                           selected=selected, unknown_journey=bool(requested) and requested != selected,
                           threshold_c=_default_threshold(), warning_margin_c=WARNING_MARGIN_C,
                           min_history_min=MIN_HISTORY_MIN, recent_readings=RECENT_READINGS,
                           card=card)


# --- JSON API --------------------------------------------------------------


@bp.get("/api/journeys")
def api_journeys():
    try:
        threshold = parse_threshold(request.args.get("threshold"), _default_threshold())
        journeys = _store().journey_list(threshold)
    except InvalidRequest as err:
        return _api_error(str(err), 400)
    except DataUnavailable as err:
        return _api_error(str(err), 503)
    artifact, _ = _models().get()
    if artifact:
        train, test = set(artifact["train_journeys"]), set(artifact["test_journeys"])
        for j in journeys:
            j["split"] = "train" if j["id"] in train else "test" if j["id"] in test else "unseen"
    else:
        for j in journeys:
            j["split"] = None
    return jsonify({"threshold_c": threshold, "journeys": journeys})


@bp.get("/api/journeys/<journey_id>")
def api_journey(journey_id):
    try:
        payload = journey_payload(_store(), _models(), journey_id,
                                  position_raw=request.args.get("at"),
                                  threshold_raw=request.args.get("threshold"),
                                  default_threshold=_default_threshold())
    except KeyError:
        return _api_error(f"Unknown journey: {journey_id}", 404)
    except InvalidRequest as err:
        return _api_error(str(err), 400)
    except DataUnavailable as err:
        return _api_error(str(err), 503)
    return jsonify(payload)


@bp.get("/api/model")
def api_model():
    _, error = _models().get()
    card = model_card(_models())
    return jsonify({"available": card is not None, "error": error, "model": card})


# --- Errors ----------------------------------------------------------------


@bp.app_errorhandler(404)
def not_found(_err):
    if request.path.startswith("/api/"):
        return _api_error("Not found", 404)
    return render_template("error.html", code=404, message="This page does not exist."), 404


@bp.app_errorhandler(500)
def server_error(_err):
    if request.path.startswith("/api/"):
        return _api_error("Internal server error", 500)
    return render_template("error.html", code=500, message="Something went wrong on the server."), 500
