import joblib
import numpy as np

import config
from .da_groups import GROUP_ORDER

ROLES = ("provider", "patient", "caregiver")

MODELS = [
    {"key": "speaker_roles", "label": "Speaker roles", "short": "Roles", "file": "speaker_roles.joblib", "needs": ()},
    {"key": "speaker_da", "label": "Dialogue acts", "short": "Dialogue acts", "file": "speaker_da.joblib", "needs": ("dialogue_acts",)},
]

BAND_LABELS = {"normal_to_mild": "Normal to mild", "moderate": "Moderate", "severe": "Severe"}

INTERPRETATIONS = {
    "normal_to_mild": {
        "label": "Normal to mild", "range": "0 to 18",
        "summary": "Scores in this range usually go with independence in most daily activities. Changes tend to show up in memory and in "
                   "more complex tasks such as managing money, appointments, and medications.",
        "focus": ["Memory aids and routines, such as calendars, pill organizers, and reminders",
                  "Check-ins on driving, finances, and medication management",
                  "Staying socially and physically active",
                  "Early planning: legal papers and care preferences"],
    },
    "moderate": {
        "label": "Moderate", "range": "19 to 36",
        "summary": "Scores in this range usually mean help is needed with many everyday tasks, including household and sometimes personal care. "
                   "Conversation may be harder to follow, questions may repeat, and caregivers often speak for the patient.",
        "focus": ["Caregiver support, respite, and shared decision making",
                  "Supervising medications and managing money",
                  "Home safety, wandering, and driving",
                  "Managing mood and behavior changes",
                  "Advance care planning while the patient can still take part"],
    },
    "severe": {
        "label": "Severe", "range": "37 to 54",
        "summary": "Scores in this range usually mean dependence on others for most daily activities, such as bathing, dressing, eating, and "
                   "continence. Speech is often very limited and the caregiver provides most of the history.",
        "focus": ["Comfort, dignity, and quality of life",
                  "Caregiver strain and available support",
                  "Safety, nutrition, swallowing, and skin care",
                  "Goals-of-care, palliative care, and hospice conversations"],
    },
}


def band(score):
    return next(name for name, upper in config.BANDS if score <= upper + 0.49)


def role_features(df):
    counts = df["role"].value_counts()
    return {f"{role}_pct": 100 * int(counts.get(role, 0)) / len(df) for role in ROLES}


def da_features(da):
    total = sum(da["role_totals"].values())
    return {f"{role}_{group}_pct": 100 * da["role_group_counts"][role].get(group, 0) / total if total else 0.0
            for role in ROLES for group in GROUP_ORDER}


def describe(name):
    role, group = name[:-4].split("_", 1) if name.count("_") > 1 else (name[:-4], None)
    if group is None:
        return f"{role.title()} share of turns", "of turns"
    
    return f"{role.title()}: {group.replace('_', ' ')}", "of dialogue act segments"


def load_artifact(path):
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        artifact = {"model": artifact, "features": list(getattr(artifact, "feature_names_in_", []))}

    if not artifact.get("features"):
        raise ValueError("saved model has no feature names")
    
    return artifact


def parts(model):
    steps = getattr(model, "steps", None)
    if steps is None:
        return None, model
    
    scaler = next((s for _, s in steps if hasattr(s, "mean_") and hasattr(s, "scale_")), None)
    return scaler, steps[-1][1]


def explain(artifact, x):
    model, names = artifact["model"], artifact["features"]
    scaler, final = parts(model)
    coef = getattr(final, "coef_", None)
    if coef is None:
        return None
    
    coef = np.ravel(coef)
    means = scaler.mean_ if scaler is not None else np.asarray(artifact.get("feature_means", np.zeros(len(names))), dtype=float)
    scale = scaler.scale_ if scaler is not None else np.ones(len(names))
    points = coef * (x - means) / scale
    baseline = float(model.predict(means.reshape(1, -1))[0])
    contributions = []
    for i in np.argsort(-np.abs(points)):
        if points[i] != 0 and len(contributions) < 8:
            label, unit = describe(names[i])
            contributions.append({"label": label, "unit": unit, "value": round(float(x[i]), 1),
                                  "average": round(float(means[i]), 1), "points": round(float(points[i]), 2)})
    
    return {"baseline": round(baseline, 1), "contributions": contributions}


def run_model(spec, available, results):
    out = {"key": spec["key"], "label": spec["label"], "short": spec["short"]}
    path = config.MODEL_DIR / spec["file"]

    if not path.exists():
        return {**out, "status": "missing", "message": f"Model not saved yet (expected {path})."}
    if any(need not in results for need in spec["needs"]):
        return {**out, "status": "blocked", "message": "Needs the dialogue act step to finish first."}
    
    try:
        artifact = load_artifact(path)
        names = artifact["features"]
        absent = [n for n in names if n not in available]
        if absent:
            raise KeyError(f"features not available for this visit: {absent[:3]}")
        x = np.array([available[n] for n in names], dtype=float)
        score = min(max(float(artifact["model"].predict(x.reshape(1, -1))[0]), 0.0), float(config.DSRS_MAX))
        out.update(status="ok", predicted=round(score, 1), band=band(score), explanation=explain(artifact, x))
        return out
    except Exception as error:
        return {**out, "status": "error", "message": f"{type(error).__name__}: {error}"}


def run(visit, df, results):
    available = dict(role_features(df))
    if "dialogue_acts" in results:
        available.update(da_features(results["dialogue_acts"]))

    models = [run_model(spec, available, results) for spec in MODELS]
    ok = [m for m in models if m["status"] == "ok"]
    out = {"models": models, "band_labels": BAND_LABELS, "interpretations": INTERPRETATIONS, "status": "ok" if ok else "no_models"}
    if config.SHOW_REFERENCE_SCORE and visit["meta"].get("assumed_dsrs") is not None:
        out["reference_score"] = visit["meta"]["assumed_dsrs"]
    if not ok:
        return out
    average = sum(m["predicted"] for m in ok) / len(ok)
    
    out["summary"] = {"predicted": round(average, 1), "band": band(average), "n_models": len(ok)}
    return out
