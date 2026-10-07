from .text_utils import normalize_ar


# ============== قراءة المواقع ==============
def _first(d: dict, keys: tuple[str, ...]) -> str:
    return next((str(d[k]).strip() for k in keys if d.get(k)), "")


def to_location(d: dict, fallback_id: str = "") -> dict | None:
    if not isinstance(d, dict):
        return None
    c = d.get("coordinates") if isinstance(d.get("coordinates"), dict) else d
    try:
        lat = float(c.get("lat", c.get("latitude")))
        lng = float(c.get("lng", c.get("lon", c.get("longitude"))))
    except (TypeError, ValueError):
        return None
    name = _first(d, ("name", "title", "اسم"))
    if not (-90 <= lat <= 90 and -180 <= lng <= 180) or not name:
        return None
    link = _first(d, ("maps_link", "map_link", "map_url", "url"))
    return {"id": str(d.get("id") or fallback_id), "name": name, "lat": lat, "lng": lng, "maps_link": link}


def short_title(name: str) -> str:
    return (name.split("|")[0].strip() or name.strip())[:100]


# ============== مطابقة المكان ==============
def _match_names(name: str) -> list[str]:
    variants = (v for part in name.split("|") for v in (part.strip(), part.strip().split(" - ")[0].strip()))
    return list(dict.fromkeys(n for n in map(normalize_ar, variants) if len(n) >= 6))


def resolve_location(text: str, locs: list[dict]) -> dict | None:
    norm = normalize_ar(text or "")
    candidates = [((i, -len(n)), loc) for loc in locs for n in _match_names(loc["name"]) if (i := norm.find(n)) >= 0]
    return min(candidates, key=lambda c: c[0])[1] if candidates else None
