"""Tests for the FrostIQ Flask website and its JSON API."""

import json

import numpy as np
import pandas as pd
import pytest

from app import create_app
from app.services import (
    LEVEL_ABOVE,
    LEVEL_NORMAL,
    LEVEL_WARNING,
    InvalidRequest,
    parse_position,
    parse_threshold,
    status_level,
)
from frostiq.features import build_features
from frostiq.model import predict_latest, save_model, train_model
from frostiq.simulate import SimulationConfig, add_threshold_labels, generate_dataset, save_dataset

CFG = SimulationConfig(n_journeys=30, seed=9)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    folder = tmp_path_factory.mktemp("web")
    readings, journeys = generate_dataset(CFG)
    paths = save_dataset(readings, journeys, CFG, 8.0, folder)
    artifact = train_model(readings)
    model_path = save_model(artifact, folder / "model.joblib")
    return {"readings": paths["readings"], "model": model_path, "artifact": artifact,
            "data": pd.read_csv(paths["readings"], parse_dates=["timestamp"])}


@pytest.fixture(scope="module")
def client(files):
    return create_app(readings_path=files["readings"], model_path=files["model"]).test_client()


def _strict_json(response):
    """Parse JSON and fail on NaN/Infinity, which browsers cannot parse."""
    def reject(value):
        raise AssertionError(f"non-standard JSON constant {value}")
    return json.loads(response.get_data(as_text=True), parse_constant=reject)


def _rows(files, journey_id):
    return files["data"][files["data"]["journey_id"] == journey_id].reset_index(drop=True)


def _find(files, condition):
    for journey_id, rows in files["data"].groupby("journey_id"):
        rows = rows.reset_index(drop=True)
        for i in range(len(rows)):
            if condition(rows, i):
                return journey_id, i
    raise AssertionError("no matching reading in test data")


# --- Pages and navigation ----------------------------------------------------


STORY_SECTIONS = ("home", "problem", "journey", "risk", "how", "predict", "performance", "limits")


def _hero(html):
    return html[html.index('id="home"'):html.index('id="problem"')]


def _section(html, name, next_name):
    return html[html.index(f'id="{name}"'):html.index(f'id="{next_name}"')]


def test_landing_hero(client):
    r = client.get("/")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    hero = _hero(html)
    assert 'Every vaccine has a <span class="accent">journey.</span>' in hero
    assert "Every degree matters." in hero
    assert "FrostIQ explores how machine learning can help identify temperature risks" in hero
    assert 'href="#problem">Explore the story' in hero
    assert 'href="/dashboard">Open Monitoring Console' in hero
    assert "Not validated for, and not to be used with, real medicines or vaccines" in hero
    # No model-performance statistics in the hero.
    assert "recall" not in hero.lower() and "precision" not in hero.lower()


def test_story_sections_exist_in_order_with_3d_scenes(client):
    html = client.get("/").get_data(as_text=True)
    positions = [html.index(f'id="{s}" data-scene="{s}"') for s in STORY_SECTIONS]
    assert positions == sorted(positions)


def test_navigation_links_point_to_existing_targets(client):
    html = client.get("/").get_data(as_text=True)
    for anchor in ("home", "problem", "journey", "how", "predict", "performance", "limits"):
        assert f'href="/#{anchor}"' in html and f'id="{anchor}"' in html
    assert 'href="/dashboard"' in html and ">Monitoring Console</a>" in html
    assert ">Open full monitoring console" in html and ">Open the Monitoring Console" in html


def test_homepage_loads_vendored_3d_libraries(client):
    html = client.get("/").get_data(as_text=True)
    assert '<canvas class="stage" id="stage"' in html
    assert '"three": "/static/vendor/three/three.module.min.js"' in html
    assert '"three/addons/": "/static/vendor/three/"' in html
    assert "/static/vendor/gsap/gsap.min.js" in html and "/static/vendor/gsap/ScrollTrigger.min.js" in html
    assert 'type="module" src="/static/js/story3d.js"' in html
    assert "cdn.jsdelivr.net" not in html  # the homepage needs no CDN for its 3D


def test_static_and_vendored_assets_are_served(client):
    for path in ("/static/css/frostiq.css", "/static/css/story.css", "/static/css/console.css",
                 "/static/js/nav.js", "/static/js/story.js", "/static/js/story3d.js",
                 "/static/js/stage/frostiq-stage.js", "/static/js/prediction-centre.js",
                 "/static/js/journey-story.js", "/static/js/dashboard.js",
                 "/static/vendor/three/three.module.min.js", "/static/vendor/three/RoundedBoxGeometry.js",
                 "/static/vendor/three/LICENSE", "/static/vendor/gsap/gsap.min.js",
                 "/static/vendor/gsap/ScrollTrigger.min.js", "/static/vendor/gsap/package.json",
                 "/static/vendor/README.md"):
        assert client.get(path).status_code == 200, path


def test_vendored_licences_are_included(client):
    assert "MIT License" in client.get("/static/vendor/three/LICENSE").get_data(as_text=True)
    assert "gsap.com/standard-license" in client.get("/static/vendor/gsap/package.json").get_data(as_text=True)


def test_fallback_visual_exists_for_no_webgl(client):
    hero = _hero(client.get("/").get_data(as_text=True))
    assert 'class="fallback-only"' in hero and 'id="hero-cooler"' in hero


def test_landing_shows_real_dataset_numbers(client, files):
    journey = _section(client.get("/").get_data(as_text=True), "journey", "risk")
    assert f"{CFG.n_journeys:,} simulated journeys" in journey
    assert f"{len(files['data']):,} recorded readings" in journey


def test_predict_section_uses_the_api_and_two_cards(client):
    predict = _section(client.get("/").get_data(as_text=True), "predict", "performance")
    assert 'id="journey-story"' in predict
    assert 'data-api-journeys="/api/journeys"' in predict and 'data-api-journey="/api/journeys/__ID__"' in predict
    assert "simulated" in predict
    assert predict.count("data-pc-card=") == 2


def _prediction_centre_checks(block):
    assert block.count("data-pc-card=") == 2
    cls = block[block.index('data-pc-card="classification"'):block.index('data-pc-card="regression"')]
    reg = block[block.index('data-pc-card="regression"'):]
    assert "<h3>Classification</h3>" in cls and "Risk decision" in cls
    assert "Will the temperature cross 8°C within the next 120 minutes?" in cls
    assert "FLAGGED / NOT FLAGGED" in cls
    assert "Predicts whether a threshold crossing is likely within the prediction horizon." in cls
    assert "RandomForestClassifier" in cls
    assert "<h3>Regression</h3>" in reg and "Numeric estimate" in reg and "Experimental" in reg
    assert "If a crossing is estimated, how many minutes until it happens?" in reg
    assert "Estimated minutes" in reg
    assert "Predicts a numerical estimate of time to breach." in reg
    assert "not a guaranteed warning time" in reg
    assert "HistGradientBoostingRegressor" in reg
    assert "answers a yes/no risk question" in block and "estimates a number" in block
    # No hardcoded result: both values start empty and are filled from the API.
    assert block.count("data-pc-value>—</div>") == 2


def test_prediction_centre_cards_on_homepage(client):
    html = client.get("/").get_data(as_text=True)
    _prediction_centre_checks(html[html.index('id="home-pc"'):html.index('id="js-error"')])


def test_prediction_centre_cards_on_console(client):
    html = client.get("/dashboard").get_data(as_text=True)
    _prediction_centre_checks(html[html.index('id="console-pc"'):html.index('id="temp-chart"')])


def test_performance_section_labels_metric_populations(client, files):
    html = client.get("/").get_data(as_text=True)
    perf = _section(html, "performance", "limits")
    m = files["artifact"]["held_out_metrics"]
    assert f"{m['recall']:.2f}" in perf and f"{m['precision']:.2f}" in perf
    for key in ("tp", "fn", "fp", "tn"):
        assert f"<b>{m[key]:,}</b>" in perf
    crossing_row = perf[perf.index("Minutes error on crossing cases"):]
    assert f"{m['mae_crossing']:.1f} min" in crossing_row.split("</tr>")[0]
    all_row = perf[perf.index("Minutes error, all readings"):]
    assert f"{m['mae_all']:.1f} min" in all_row.split("</tr>")[0]
    assert "not</em> the error on crossing cases" in all_row.split("</tr>")[0]
    counts = html[html.index('id="model-counts"'):].split("</script>")[0]
    assert f'"tp": {m["tp"]}' in counts and f'"tn": {m["tn"]}' in counts


def test_templates_reload_without_server_restart(files):
    # A long-running server must pick up template edits (the cause of a stale homepage once).
    app = create_app(readings_path=files["readings"], model_path=files["model"])
    assert app.config["TEMPLATES_AUTO_RELOAD"] is True
    assert app.jinja_env.auto_reload is True


def test_repository_link_is_the_verified_remote(client):
    html = client.get("/").get_data(as_text=True)
    assert 'href="https://github.com/PrachiPangire/FrostIQ"' in html


def test_dashboard_renders_console(client):
    r = client.get("/dashboard")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    for element in ('id="journey-select"', 'id="temp-chart"', 'id="position-slider"', 'id="threshold-form"',
                    'id="console-pc"', 'id="recent-body"', "Status rules", "not live telemetry",
                    'id="empty-state"', 'id="journey-view"', 'id="threshold-note"', 'id="now-journey"',
                    'id="hindsight-toggle"', 'id="play"', 'id="step-back"', 'id="step-fwd"', 'id="to-latest"',
                    "/static/js/prediction-centre.js"):
        assert element in html, element
    assert 'data-selected="J0001"' in html
    assert "chart.js@4.4.1" in html and 'integrity="sha384-' in html


def test_dashboard_layout_puts_observed_before_predicted_before_details(client):
    html = client.get("/dashboard").get_data(as_text=True)
    order = [html.index(s) for s in ('id="now-journey"', 'id="latest-temp"', 'id="status-level"',
                                      'id="console-pc"', 'id="temp-chart"', 'id="recent-body"')]
    assert order == sorted(order)


def test_chart_fallback_does_not_overpromise(client):
    html = client.get("/dashboard").get_data(as_text=True)
    fallback = html[html.index('id="chart-fallback"'):]
    fallback = fallback[:fallback.index("</div>")]
    assert "All values remain available" not in fallback
    assert "latest 12 readings" in fallback


def test_dashboard_preselects_requested_journey(client):
    html = client.get("/dashboard?journey=J0005").get_data(as_text=True)
    assert 'data-selected="J0005"' in html


def test_dashboard_with_unknown_journey_falls_back(client):
    html = client.get("/dashboard?journey=NOPE").get_data(as_text=True)
    assert 'data-selected="J0001"' in html and "does not exist" in html


def test_unknown_page_is_404(client):
    r = client.get("/no-such-page")
    assert r.status_code == 404 and "404" in r.get_data(as_text=True)


# --- Journey list ------------------------------------------------------------


def test_journey_list(client, files):
    body = _strict_json(client.get("/api/journeys"))
    assert body["threshold_c"] == 8.0
    assert len(body["journeys"]) == CFG.n_journeys
    max_temp = files["data"].groupby("journey_id")["temperature_c"].max()
    for j in body["journeys"]:
        assert j["crossed"] == bool(max_temp[j["id"]] > 8.0)
        assert j["split"] in {"train", "test"}
        assert j["readings"] == len(_rows(files, j["id"]))


def test_journey_list_follows_threshold(client, files):
    body = _strict_json(client.get("/api/journeys?threshold=30"))
    max_temp = files["data"].groupby("journey_id")["temperature_c"].max()
    assert [j["crossed"] for j in body["journeys"]] == [bool(max_temp[j["id"]] > 30) for j in body["journeys"]]


# --- One journey: data, chart, selection --------------------------------------


def test_journey_payload_uses_recorded_values_only(client, files):
    rows = _rows(files, "J0002")
    body = _strict_json(client.get("/api/journeys/J0002"))
    assert body["position"] == len(rows) - 1 == body["total_readings"] - 1
    np.testing.assert_allclose(body["chart"]["temperature_c"], rows["temperature_c"].round(2))
    assert body["chart"]["future_temperature_c"] == []
    assert body["latest"]["temperature_c"] == pytest.approx(rows["temperature_c"].iloc[-1])
    assert body["latest"]["margin_c"] == pytest.approx(8.0 - rows["temperature_c"].iloc[-1], abs=0.01)
    assert body["summary"]["max_c"] == pytest.approx(rows["temperature_c"].max(), abs=0.01)
    assert body["scenario"] == rows["scenario"].iloc[0]


def test_replay_position_limits_visible_history(client, files):
    rows = _rows(files, "J0003")
    body = _strict_json(client.get("/api/journeys/J0003?at=40"))
    assert len(body["chart"]["temperature_c"]) == 41
    assert len(body["chart"]["future_temperature_c"]) == len(rows) - 41
    assert body["summary"]["readings"] == 41
    assert body["latest"]["temperature_c"] == pytest.approx(rows["temperature_c"][40])
    recent = body["recent"]
    assert len(recent) == 12 and recent[0]["temperature_c"] == pytest.approx(rows["temperature_c"][40])
    assert [r["elapsed_min"] for r in recent] == sorted((r["elapsed_min"] for r in recent), reverse=True)


def test_selecting_different_journeys_changes_data(client):
    a = _strict_json(client.get("/api/journeys/J0001"))
    b = _strict_json(client.get("/api/journeys/J0002"))
    assert a["journey_id"] == "J0001" and b["journey_id"] == "J0002"
    assert a["chart"]["temperature_c"] != b["chart"]["temperature_c"]


def test_hindsight_matches_labels(client, files):
    journey_id, i = _find(files, lambda r, i: i >= 6 and r["temperature_c"][i] <= 8.0
                          and (r["temperature_c"][i + 1:] > 8.0).any())
    body = _strict_json(client.get(f"/api/journeys/{journey_id}?at={i}"))
    labelled = add_threshold_labels(_rows(files, journey_id)[["journey_id", "timestamp", "temperature_c"]], 8.0)
    assert body["hindsight"]["outcome"] == "crossed_later"
    assert body["hindsight"]["minutes_to_crossing"] == pytest.approx(labelled["minutes_to_threshold"][i])


# --- Predictions -------------------------------------------------------------


def test_prediction_matches_model_on_visible_history(client, files):
    journey_id, i = _find(files, lambda r, i: i >= 20 and r["temperature_c"][i] <= 8.0)
    body = _strict_json(client.get(f"/api/journeys/{journey_id}?at={i}"))
    visible = _rows(files, journey_id).iloc[: i + 1]
    expected = predict_latest(files["artifact"], build_features(visible[["journey_id", "timestamp", "temperature_c"]]), 8.0)
    p = body["prediction"]
    assert p["status"] == "predicted"
    assert p["minutes"] == pytest.approx(round(expected["minutes"], 1))
    assert p["crossing_score"] == pytest.approx(round(expected["crossing_score"], 3))
    assert body["split"] in {"train", "test"}


def test_observed_crossing_is_not_a_prediction(client, files):
    journey_id, i = _find(files, lambda r, i: i >= 10 and r["temperature_c"][i] > 8.0)
    body = _strict_json(client.get(f"/api/journeys/{journey_id}?at={i}"))
    assert body["status"]["level"] == LEVEL_ABOVE
    assert body["prediction"]["status"] == "observed_above"
    assert body["prediction"]["minutes"] is None


def test_insufficient_history_gives_no_prediction(client, files):
    journey_id, i = _find(files, lambda r, i: i == 2 and r["temperature_c"][i] <= 8.0)
    body = _strict_json(client.get(f"/api/journeys/{journey_id}?at={i}"))
    assert body["prediction"]["status"] == "insufficient_history"
    assert body["prediction"]["minutes"] is None


def test_other_threshold_disables_prediction_but_keeps_status_rules(client):
    body = _strict_json(client.get("/api/journeys/J0001?threshold=25&at=30"))
    assert body["threshold_c"] == 25.0
    assert body["prediction"]["status"] == "threshold_mismatch"
    assert body["status"]["level"] in {LEVEL_NORMAL, LEVEL_WARNING, LEVEL_ABOVE}
    assert body["latest"]["margin_c"] == pytest.approx(25.0 - body["latest"]["temperature_c"], abs=0.01)


# --- The two AI outputs: classification and regression -------------------------


def _first_with(client, files, wanted):
    """Find a (journey, position) whose API prediction satisfies `wanted`."""
    for journey_id, rows in files["data"].groupby("journey_id"):
        for i in range(6, len(rows), 3):
            body = _strict_json(client.get(f"/api/journeys/{journey_id}?at={i}"))
            if wanted(body):
                return body
    raise AssertionError("no matching prediction in test data")


def test_classification_card_matches_classifier_output(client, files):
    body = _first_with(client, files, lambda b: b["prediction"]["status"] == "predicted")
    c, p = body["classification"], body["prediction"]
    assert c["model"] == "RandomForestClassifier" and c["available"] and c["status"] == "predicted"
    assert c["score"] == p["crossing_score"]
    assert c["flagged"] == (p["crossing_score"] >= 0.5) == p["flags_crossing"]
    assert c["decision_threshold"] == 0.5 and c["horizon_min"] == 120.0


def test_regression_card_reports_real_estimate(client, files):
    body = _first_with(client, files, lambda b: b["prediction"]["status"] == "predicted" and not b["prediction"]["capped"])
    r, p = body["regression"], body["prediction"]
    assert r["model"] == "HistGradientBoostingRegressor" and r["experimental"]
    assert r["status"] == "estimated" and r["minutes"] == p["minutes"] == r["model_output_min"]


def test_regression_at_horizon_cap_is_not_a_crossing_time(client, files):
    body = _first_with(client, files, lambda b: b["prediction"]["status"] == "predicted" and b["prediction"]["capped"])
    r = body["regression"]
    assert r["status"] == "beyond_horizon" and r["minutes"] is None
    assert r["model_output_min"] == body["prediction"]["minutes"]
    assert "not a predicted crossing time" in r["message"]


def test_prediction_cards_unavailable_states(client, files):
    above = _first_with(client, files, lambda b: b["prediction"]["status"] == "observed_above")
    for card in (above["classification"], above["regression"]):
        assert card["available"] is False and card["status"] == "observed_above"
    assert above["classification"]["flagged"] is None and above["regression"]["minutes"] is None

    journey_id, i = _find(files, lambda r, i: i == 2 and r["temperature_c"][i] <= 8.0)
    early = _strict_json(client.get(f"/api/journeys/{journey_id}?at={i}"))
    assert early["classification"]["status"] == early["regression"]["status"] == "insufficient_history"

    other = _strict_json(client.get("/api/journeys/J0001?at=30&threshold=25"))
    assert other["classification"]["status"] == other["regression"]["status"] == "threshold_mismatch"
    assert other["classification"]["score"] is None and other["regression"]["minutes"] is None


def test_prediction_cards_without_model(files, tmp_path):
    c = create_app(readings_path=files["readings"], model_path=tmp_path / "none.joblib").test_client()
    body = _strict_json(c.get("/api/journeys/J0001?at=30"))
    for card in (body["classification"], body["regression"]):
        assert card["available"] is False and card["status"] == "model_unavailable"
        assert "python -m frostiq.model train" in card["message"]


def test_model_card_has_confusion_matrix(client, files):
    m = _strict_json(client.get("/api/model"))["model"]
    assert m["tp"] + m["fn"] == m["positives"] and m["fp"] + m["tn"] == m["negatives"]
    assert m["tp"] == m["caught"] and m["fn"] == m["missed"]


def test_model_endpoint(client, files):
    body = _strict_json(client.get("/api/model"))
    assert body["available"] and body["error"] is None
    assert body["model"]["train_journeys"] == len(files["artifact"]["train_journeys"])


# --- Status rules and input parsing ------------------------------------------


def test_status_rules():
    assert status_level(9.0, 8.0, None)["level"] == LEVEL_ABOVE
    assert status_level(8.0, 8.0, None)["level"] == LEVEL_WARNING  # exactly at threshold is not above
    assert status_level(7.2, 8.0, None)["level"] == LEVEL_WARNING  # margin 0.8 <= 1.0
    assert status_level(5.0, 8.0, None)["level"] == LEVEL_NORMAL
    flagged = {"status": "predicted", "flags_crossing": True, "horizon_min": 120.0}
    assert status_level(5.0, 8.0, flagged)["level"] == LEVEL_WARNING
    assert status_level(5.0, 8.0, {**flagged, "flags_crossing": False})["level"] == LEVEL_NORMAL


def test_parse_inputs():
    assert parse_threshold(None, 8.0) == 8.0 and parse_threshold("9.5", 8.0) == 9.5
    assert parse_position(None, 10) == 9 and parse_position("0", 10) == 0
    for bad in ("abc", "nan", "inf", "100", "-31"):
        with pytest.raises(InvalidRequest):
            parse_threshold(bad, 8.0)
    for bad in ("abc", "-1", "10", "2.5"):
        with pytest.raises(InvalidRequest):
            parse_position(bad, 10)


@pytest.mark.parametrize("url, status", [
    ("/api/journeys/NOPE", 404),
    ("/api/journeys/J0001?at=-1", 400),
    ("/api/journeys/J0001?at=abc", 400),
    ("/api/journeys/J0001?at=99999", 400),
    ("/api/journeys/J0001?threshold=abc", 400),
    ("/api/journeys/J0001?threshold=inf", 400),
    ("/api/journeys?threshold=999", 400),
    ("/api/nothing-here", 404),
])
def test_invalid_requests_return_json_errors(client, url, status):
    r = client.get(url)
    assert r.status_code == status
    assert "error" in _strict_json(r)


# --- Missing data, missing model, server errors ------------------------------


def test_missing_dataset_is_handled(files, tmp_path):
    c = create_app(readings_path=tmp_path / "none.csv", model_path=files["model"]).test_client()
    assert c.get("/").status_code == 200
    assert "python -m frostiq.simulate" in c.get("/").get_data(as_text=True)
    html = c.get("/dashboard").get_data(as_text=True)
    assert "Dataset unavailable" in html and 'id="journey-select"' not in html
    for url in ("/api/journeys", "/api/journeys/J0001"):
        r = c.get(url)
        assert r.status_code == 503 and "python -m frostiq.simulate" in r.get_json()["error"]


def test_unreadable_dataset_is_handled(files, tmp_path):
    bad = tmp_path / "synthetic_readings.csv"
    bad.write_text("not,a,valid\nreadings,file,here\n", encoding="utf-8")
    c = create_app(readings_path=bad, model_path=files["model"]).test_client()
    assert c.get("/api/journeys").status_code == 503


@pytest.mark.parametrize("make_model", ["missing", "corrupted"])
def test_missing_or_corrupted_model_is_handled(files, tmp_path, make_model):
    model_path = tmp_path / "model.joblib"
    if make_model == "corrupted":
        model_path.write_bytes(b"garbage")
    c = create_app(readings_path=files["readings"], model_path=model_path).test_client()
    body = _strict_json(c.get("/api/journeys/J0001?at=30"))
    assert body["prediction"] is None and "python -m frostiq.model train" in body["model_error"]
    assert body["status"]["level"] in {LEVEL_NORMAL, LEVEL_WARNING, LEVEL_ABOVE}  # rules still work
    assert body["split"] is None
    assert _strict_json(c.get("/api/model"))["available"] is False
    assert all(j["split"] is None for j in _strict_json(c.get("/api/journeys"))["journeys"])
    assert "Predictions unavailable" in c.get("/dashboard").get_data(as_text=True)
    assert "not trained" in c.get("/").get_data(as_text=True)


def test_unexpected_server_error_returns_json_500(files, monkeypatch):
    app = create_app(readings_path=files["readings"], model_path=files["model"])
    store = app.extensions["frostiq_store"]
    monkeypatch.setattr(store, "journey", lambda _id: (_ for _ in ()).throw(RuntimeError("boom")))
    r = app.test_client().get("/api/journeys/J0001")
    assert r.status_code == 500 and r.get_json() == {"error": "Internal server error"}
