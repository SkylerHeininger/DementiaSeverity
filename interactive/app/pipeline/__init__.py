import json
import math
import threading
import time
from datetime import datetime

import config
import visits

from . import dialogue_acts, severity, stats

STEPS = [
    ("stats", "Visit summary statistics", stats.run),
    ("dialogue_acts", "Dialogue acts", dialogue_acts.run),
    ("severity", "DSRS prediction", severity.run),
]

SCHEMA = 3
jobs = {}
lock = threading.Lock()


def results_path(visit_id):
    return config.PROCESSED_DIR / f"{visit_id}.json"


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def load_results(visit_id):
    path = results_path(visit_id)
    return clean(json.loads(path.read_text())) if path.exists() else None


def state(visit_id):
    saved = load_results(visit_id)
    if saved is None:
        return "none"
    return "partial" if any(s["status"] == "error" for s in saved["steps"].values()) else "done"


def save_results(visit_id, results, steps):
    previous = load_results(visit_id)
    if not previous or previous.get("schema") != SCHEMA:
        previous = {"results": {}}
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results_path(visit_id).write_text(json.dumps(
        clean({"schema": SCHEMA, "processed_at": datetime.now().isoformat(timespec="seconds"), "steps": steps, "results": {**previous["results"], **results}}),
        default=lambda o: clean(o.item()) if hasattr(o, "item") else str(o)))


def status(visit_id):
    with lock:
        job = jobs.get(visit_id)
        return json.loads(json.dumps(job)) if job else {"running": False, "finished": False, "steps": {}}


def start(visits):
    with lock:
        todo = [v for v in visits if not jobs.get(v["id"], {}).get("running")]
        for visit in todo:
            jobs[visit["id"]] = {
                "running": True, "finished": False,
                "steps": {key: {"label": label, "status": "pending", "message": "", "seconds": None} for key, label, _ in STEPS},
            }
    if todo:
        threading.Thread(target=run_visits, args=(todo,), daemon=True).start()
    return [v["id"] for v in todo]


def run_visits(visits):
    for visit in visits:
        run_steps(visit)


def run_steps(visit):
    results = {}
    try:
        df = visits.load_frame(visit)
    except Exception as error:
        df, load_error = None, f"{type(error).__name__}: {error}"
    for key, _, fn in STEPS:
        with lock:
            step = jobs[visit["id"]]["steps"][key]
            step["status"] = "running"
        began = time.time()
        try:
            if df is None:
                raise RuntimeError(load_error)
            results[key] = fn(visit, df, results)
            outcome = {"status": "done"}
        except Exception as error:
            outcome = {"status": "error", "message": f"{type(error).__name__}: {error}"}
        with lock:
            step.update(outcome, seconds=round(time.time() - began, 1))
    with lock:
        save_results(visit["id"], results, jobs[visit["id"]]["steps"])
        jobs[visit["id"]].update(running=False, finished=True)
