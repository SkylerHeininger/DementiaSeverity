import os
import sys

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

import config
import pipeline
import visits
from pipeline import view
from config import ROLES

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
app.json.sort_keys = False
app.jinja_env.policies["json.dumps_kwargs"] = {"sort_keys": False}


@app.url_defaults
def version_static(endpoint, values):
    if endpoint == "static" and "filename" in values:
        path = config.ROOT / "static" / values["filename"]
        if path.exists():
            values["v"] = int(path.stat().st_mtime)


ROLE_LABEL = {"provider": "Provider", "patient": "Patient", "caregiver": "Caregiver"}
app.jinja_env.globals.update(ROLES=ROLES, ROLE_LABEL=ROLE_LABEL, MODEL_COLORS=view.MODEL_COLORS)


@app.template_filter("cap")
def cap(text):
    """Capitalize the first letter only, so acronyms like "BP" survive (Jinja's capitalize lowercases the rest)."""
    text = str(text)
    return text[:1].upper() + text[1:]


@app.template_filter("clock")
def clock(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


@app.template_filter("num")
def num(value):
    return int(value) if float(value).is_integer() else value


@app.template_filter("filesize")
def filesize(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1024 / 1024:.1f} MB"


def visit_or_404(visit_id):
    return visits.get_visit(visit_id) or abort(404)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/clinician")
def clinician():
    return render_template("upload.html", error=None)


@app.post("/upload")
def upload():
    file = request.files.get("transcript")
    if file is None or not file.filename:
        return render_template("upload.html", error="Choose a CSV file to upload."), 400

    try:
        visit = visits.save_upload(file)
    except ValueError as error:
        return render_template("upload.html", error=str(error)), 400

    pipeline.start([visit])
    return redirect(url_for("visit_page", visit_id=visit["id"]))


@app.errorhandler(413)
def too_large(error):
    return render_template("upload.html", error=f"The file is larger than {config.MAX_UPLOAD_MB} MB."), 413


@app.route("/visits/<visit_id>")
def visit_page(visit_id):
    visit = visit_or_404(visit_id)
    saved = pipeline.load_results(visit_id)
    outdated = bool(saved) and saved.get("schema") != pipeline.SCHEMA
    if outdated:
        saved = None
    results = saved["results"] if saved else {}
    steps = saved["steps"] if saved else {}
    status = pipeline.status(visit_id)
    pending = {key: {"label": label, "status": "pending"} for key, label, _ in pipeline.STEPS}
    payload = {"visit_id": visit_id, "running": status["running"], "steps": status["steps"] if status["running"] else steps or pending}
    return render_template("visit.html", visit=visit, turns=visits.turns_payload(visits.load_frame(visit)),
                           results=results, steps=steps, saved=saved, outdated=outdated, view=view.build(results), payload=payload)


@app.post("/visits/<visit_id>/process")
def process_visit(visit_id):
    return jsonify({"started": pipeline.start([visit_or_404(visit_id)])})


@app.get("/visits/<visit_id>/status")
def visit_status(visit_id):
    visit_or_404(visit_id)
    return jsonify(pipeline.status(visit_id))


@app.get("/visits/<visit_id>/results")
def visit_results(visit_id):
    visit_or_404(visit_id)
    return jsonify(pipeline.load_results(visit_id))


if __name__ == "__main__":
    print(f"Running with {sys.executable}")
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5001)), threaded=True)
