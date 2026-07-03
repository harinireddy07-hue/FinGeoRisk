"""
FinGeoRisk — Actuarial Geospatial Asset Terminal (backend)
==========================================================
Actuarial financial analytics + interactive global locations, now with:
  - an auto "why this area is at risk" summary
  - colour-coded risk hotspots
  - real-time disaster monitoring (USGS earthquakes + NASA EONET)
"""

import os
import time
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder=None)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
ENV_API_KEY = os.getenv("GEMINI_API_KEY", "")
USER_AGENT = os.getenv("USER_AGENT", "FinGeoRisk/3.0 (actuarial demo)")


def clip(value, lo, hi):
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Actuarial model
# ---------------------------------------------------------------------------
def compute_actuarial_metrics(lat, lon, location_name="Selected Property"):
    """Deterministic hyper-local cat-model -> financial underwriting parameters."""
    seed = abs(hash(f"{lat:.3f},{lon:.3f}"))

    coastal = any(w in location_name.lower() for w in ("water", "coast", "beach", "bay", "harbor", "port"))
    base_flood = (seed % 75) + 10 if coastal else (seed % 55) + 5
    base_wind = ((seed >> 2) % 65) + 15
    base_wildfire = ((seed >> 4) % 80) + 5 if base_flood < 30 else (seed % 25)

    base_flood = clip(base_flood, 0, 100)
    base_wind = clip(base_wind, 0, 100)
    base_wildfire = clip(base_wildfire, 0, 100)

    total_insured_value = 1250000 + ((seed % 500) * 5000)
    composite_risk_idx = (base_flood * 0.45) + (base_wind * 0.35) + (base_wildfire * 0.20)
    annual_premium = (total_insured_value * 0.002) * (1 + (composite_risk_idx / 25.0))
    eml_pct = clip(composite_risk_idx * 1.1, 10.0, 95.0)
    estimated_max_loss = total_insured_value * (eml_pct / 100.0)
    reward_pool_index = clip(10.0 - (composite_risk_idx / 10.0), 1.2, 9.8)

    if composite_risk_idx >= 70:
        tier = "CRITICAL EXPOSURE"
    elif composite_risk_idx >= 45:
        tier = "HIGH EXPOSURE"
    elif composite_risk_idx >= 25:
        tier = "MODERATE"
    else:
        tier = "MINIMAL"

    # ----- auto "why is this area at risk" summary -----
    perils = {"flood": base_flood, "hurricane/wind": base_wind, "wildfire": base_wildfire}
    dominant = max(perils, key=perils.get)
    tier_word = tier.replace(" EXPOSURE", "").title()
    summary = (
        f"{location_name.split(',')[0]} sits in a {tier_word} risk tier, "
        f"driven primarily by {dominant} ({perils[dominant]:.0f}% modelled payout probability). "
        f"On a ${total_insured_value:,.0f} asset, a worst-case event could destroy "
        f"{eml_pct:.0f}% of value (~${estimated_max_loss:,.0f}), so the engine targets a "
        f"${annual_premium:,.0f} annual premium. Underwriting yield rating: "
        f"{reward_pool_index:.1f}/10."
    )

    return {
        "geography": {"lat": lat, "lon": lon, "name": location_name},
        "financials": {
            "total_insured_value": round(total_insured_value, 2),
            "annual_premium": round(annual_premium, 2),
            "estimated_max_loss": round(estimated_max_loss, 2),
            "eml_pct": round(eml_pct, 1),
            "underwriting_yield": round(reward_pool_index, 1),
        },
        "vectors": {
            "flood_payout_prob": round(base_flood, 1),
            "wind_payout_prob": round(base_wind, 1),
            "wildfire_payout_prob": round(base_wildfire, 1),
            "composite_idx": round(composite_risk_idx, 1),
        },
        "tier": tier,
        "risk_summary": summary,
    }


# ---------------------------------------------------------------------------
# Risk hotspots (colour-coded on the map)
# ---------------------------------------------------------------------------
HOTSPOT_CITIES = [
    ("Miami, FL", 25.7617, -80.1918), ("New Orleans, LA", 29.9511, -90.0715),
    ("Houston, TX", 29.7604, -95.3698), ("Los Angeles, CA", 34.0522, -118.2437),
    ("San Francisco, CA", 37.7749, -122.4194), ("New York, NY", 40.7128, -74.0060),
    ("Tokyo, Japan", 35.6762, 139.6503), ("Jakarta, Indonesia", -6.2088, 106.8456),
    ("Manila, Philippines", 14.5995, 120.9842), ("Mumbai, India", 19.0760, 72.8777),
    ("Venice, Italy", 45.4408, 12.3155), ("Sydney, Australia", -33.8688, 151.2093),
]


def tier_color(tier):
    return {"CRITICAL EXPOSURE": "#ef4444", "HIGH EXPOSURE": "#fb923c",
            "MODERATE": "#fbbf24", "MINIMAL": "#34d399"}.get(tier, "#34d399")


# ---------------------------------------------------------------------------
# Live disaster feeds (cached)
# ---------------------------------------------------------------------------
_DISASTER_CACHE = {"t": 0, "data": None}
EONET_CAT = {
    "wildfires": "wildfire", "severeStorms": "storm", "floods": "flood",
    "volcanoes": "volcano", "seaLakeIce": "ice", "drought": "drought",
}


def fetch_usgs():
    out = []
    try:
        r = requests.get(
            "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson",
            headers={"User-Agent": USER_AGENT}, timeout=12)
        for f in r.json().get("features", [])[:40]:
            c = f["geometry"]["coordinates"]
            p = f["properties"]
            out.append({"type": "earthquake", "title": p.get("place", "Earthquake"),
                        "lat": c[1], "lon": c[0],
                        "detail": f"M{p.get('mag','?')}", "mag": p.get("mag"),
                        "time": p.get("time"), "source": "USGS"})
    except Exception:
        pass
    return out


def fetch_eonet():
    out = []
    try:
        r = requests.get(
            "https://eonet.gsfc.nasa.gov/api/v3/events",
            params={"status": "open", "limit": 60},
            headers={"User-Agent": USER_AGENT}, timeout=12)
        for ev in r.json().get("events", []):
            cats = ev.get("categories", [])
            cid = cats[0].get("id") if cats else ""
            etype = EONET_CAT.get(cid, "other")
            geos = ev.get("geometry", [])
            if not geos:
                continue
            g = geos[-1]                       # most recent position
            coords = g.get("coordinates")
            try:
                if g.get("type") == "Point":
                    lon, lat = coords[0], coords[1]
                elif g.get("type") == "Polygon":
                    lon, lat = coords[0][0][0], coords[0][0][1]
                else:
                    continue
            except (TypeError, IndexError):
                continue
            out.append({"type": etype, "title": ev.get("title", "Event"),
                        "lat": lat, "lon": lon,
                        "detail": cats[0].get("title", "") if cats else "",
                        "time": g.get("date"), "source": "NASA EONET"})
    except Exception:
        pass
    return out


def get_disasters():
    now = time.time()
    if _DISASTER_CACHE["data"] is not None and (now - _DISASTER_CACHE["t"]) < 300:
        return _DISASTER_CACHE["data"]

    quakes = fetch_usgs()[:35]          # cap the earthquake firehose
    natural = fetch_eonet()[:35]        # fires / storms / floods / volcanoes
    print(f"[disasters] USGS={len(quakes)}  EONET={len(natural)}")

    # interleave so neither source dominates the visible list
    merged, i, j = [], 0, 0
    while i < len(natural) or j < len(quakes):
        if i < len(natural):
            merged.append(natural[i]); i += 1
        if j < len(quakes):
            merged.append(quakes[j]); j += 1

    _DISASTER_CACHE.update({"t": now, "data": merged})
    return merged


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
def query_gemini(prompt, system_instruction, api_key):
    if not api_key:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        pass
    return None


SYSTEM_PROMPT = (
    "You are the FinGeoRisk Actuarial AI Assistant. You interpret financial-geospatial risk models.\n"
    "CRITICAL PROTOCOL: Explain the relationship between disaster probability, Estimated Maximum Loss (EML), "
    "and premium payouts based on the financial JSON context provided. Be concise and professional."
)


def offline_assessment(ctx):
    f = ctx.get("financials", {})
    v = ctx.get("vectors", {})
    if ctx.get("risk_summary"):
        return "[Offline Underwriting Intelligence] " + ctx["risk_summary"]
    return (
        f"[Offline Underwriting Intelligence] Asset value estimated at ${f.get('total_insured_value',0):,.2f} "
        f"with an Estimated Maximum Loss of {f.get('eml_pct',0)}% (${f.get('estimated_max_loss',0):,.2f}). "
        f"Target annual premium ${f.get('annual_premium',0):,.2f}; composite hazard index "
        f"{v.get('composite_idx',0)}/100; yield {f.get('underwriting_yield',0)}/10."
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/compute", methods=["POST"])
def api_compute():
    body = request.get_json(force=True) or {}
    lat = float(body.get("lat", 37.7749))
    lon = float(body.get("lon", -122.4194))
    name = body.get("name", "Selected Property Coordinate")
    return jsonify(compute_actuarial_metrics(lat, lon, name))


@app.route("/api/hotspots")
def api_hotspots():
    out = []
    for name, lat, lon in HOTSPOT_CITIES:
        m = compute_actuarial_metrics(lat, lon, name)
        out.append({"name": name, "lat": lat, "lon": lon,
                    "tier": m["tier"], "color": tier_color(m["tier"]),
                    "composite_idx": m["vectors"]["composite_idx"]})
    return jsonify(out)


@app.route("/api/disasters")
def api_disasters():
    return jsonify({"events": get_disasters()})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    body = request.get_json(force=True) or {}
    message = (body.get("message") or "").strip()
    context = body.get("context") or {}
    api_key = (body.get("api_key") or "").strip() or ENV_API_KEY
    if not message:
        return jsonify({"reply": "Enter an actuarial or economic vector query.", "mode": "offline"})
    prompt = f"Actuarial Financial Context (JSON):\n{context}\n\nUser Query: {message}"
    reply = query_gemini(prompt, SYSTEM_PROMPT, api_key)
    if reply:
        return jsonify({"reply": reply, "mode": "live"})
    return jsonify({"reply": offline_assessment(context), "mode": "offline"})


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(host=host, port=port, debug=debug)