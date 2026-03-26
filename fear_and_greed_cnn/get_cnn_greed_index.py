import json, gzip, io, time
from datetime import date, timedelta
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import requests, pandas as pd

BASE = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
H = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",

    "Accept-Encoding": "gzip, deflate",
    "Referer": "https://www.cnn.com/markets/fear-and-greed",
    "Origin": "https://www.cnn.com",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

def _urllib_get(url):
    req = Request(url, headers=H)
    with urlopen(req, timeout=20) as r:
        raw = r.read()

        if r.headers.get("Content-Encoding", "") == "gzip":
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))

def _requests_get(url):
    r = requests.get(url, headers=H, timeout=20, allow_redirects=True)
    if r.ok:
        try:
            return r.json()
        except Exception:
            pass
    data = r.content
    try:
        data = gzip.decompress(data)
    except Exception:
        pass
    try:
        txt = data.decode("utf-8", errors="ignore")
        print(f"[debug {r.status_code}] first 200 bytes:\n{txt[:200]}")
        return json.loads(txt)
    except Exception:
        raise RuntimeError(f"HTTP {r.status_code} from {url}")

def fetch_json(max_back_days=30, pause=0.6):
    for u in (BASE, BASE + "/"):
        try:
            return _urllib_get(u)
        except Exception:
            try:
                return _requests_get(u)
            except Exception:
                pass
    # 2) Fall back over recent dated URLs
    d = date.today()
    for k in range(max_back_days):
        day = (d - timedelta(days=k)).isoformat()
        for u in (f"{BASE}/{day}", f"{BASE}/{day}/"):
            try:
                return _urllib_get(u)
            except Exception:
                try:
                    return _requests_get(u)
                except Exception:
                    time.sleep(pause)
    raise RuntimeError("Unable to fetch CNN Fear & Greed data")

def to_frames(j):
    hist = pd.DataFrame(j["fear_and_greed_historical"]["data"]).rename(columns={"x":"ts","y":"score"})
    hist["date"] = pd.to_datetime(hist["ts"], unit="ms").dt.date
    hist = hist[["date","score"]].sort_values("date")

    snap = pd.json_normalize(j["fear_and_greed"])

    comps = [
        "market_momentum_sp500",
        "stock_price_strength",
        "stock_price_breadth",
        "put_call_options",
        "market_volatility",
        "safe_haven_demand",
        "junk_bond_demand",
    ]
    comp = pd.concat([pd.json_normalize(j[c]).assign(component=c) for c in comps if c in j], ignore_index=True)
    return hist, snap, comp

j = fetch_json()
hist, snap, comp = to_frames(j)

existing = pd.read_csv("cnn_fear_and_greed_index.csv")
existing["date"] = pd.to_datetime(existing["date"]).dt.date

hist.rename(columns={"score": "combined_value"}, inplace=True)

merged = pd.concat([existing, hist], ignore_index=True)
merged["combined_value"] = merged["combined_value"].round().astype(int)
merged.drop_duplicates(subset=["date"], keep="last", inplace=True)
merged.sort_values("date", inplace=True)

merged.to_csv("cnn_fear_and_greed_index.csv", index=False)
print(f"Merged data: {len(merged)} rows, {merged['date'].min()} to {merged['date'].max()}")
