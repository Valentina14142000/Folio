import time

_hits: dict[str, list[float]] = {}


def limited(key: str, max_calls: int = 20, window: float = 60.0) -> bool:
    now = time.monotonic()
    recent = [t for t in _hits.get(key, []) if now - t < window]
    recent.append(now)
    _hits[key] = recent
    if len(_hits) > 5000:
        _hits.clear()
    return len(recent) > max_calls
