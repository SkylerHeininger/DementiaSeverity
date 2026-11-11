ROLES = ("provider", "patient", "caregiver")
MODEL_COLORS = ["#0f766e", "#4f46e5", "#d97706", "#be185d"]


def stats_view(stats):
    edges = stats["timeline"]["edges"]
    return {"by": {r["role"]: r for r in stats["roles"]}, "bins": len(stats["timeline"]["seconds"][ROLES[0]]), "bin_length": edges[1] - edges[0]}


def da_view(da):
    counts = da["role_group_counts"]
    return {
        "group_totals": {g: sum(counts[r].get(g, 0) for r in ROLES) for g in da["groups"]},
        "max_cell": max([1] + [counts[r].get(g, 0) for r in ROLES for g in da["groups"]]),
        "max_questions": max([1] + [da["questions_by_role"][r]["count"] for r in ROLES]),
    }


def severity_view(severity):
    ok = [m for m in severity["models"] if m["status"] == "ok"]
    limits = {}
    for m in ok:
        explained = m.get("explanation")
        limits[m["key"]] = {"max_abs": max([0.01] + [abs(c["points"]) for c in explained["contributions"]]) if explained else 1}
    return {"ok": ok, "limits": limits}


def build(results):
    out = {}
    if "stats" in results:
        out["stats"] = stats_view(results["stats"])
    if "dialogue_acts" in results:
        out["da"] = da_view(results["dialogue_acts"])
    if "severity" in results:
        out["severity"] = severity_view(results["severity"])
    return out
