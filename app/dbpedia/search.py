import json
import time
import requests
import xmltodict
import os

API = "https://en.wikipedia.org/w/api.php"

HEADERS = {"User-Agent": "redirect-checker/1.0 (swsn42)"}

CACHE_FILE = "./link_cache2.json"
CACHE: dict[str, list] = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        CACHE = json.loads(f.read())


def wiki_result(title: str, n: int = 10):
    title = title.split("/")[-1].replace("_", " ")

    if title in CACHE:
        return CACHE[title]

    params = {"action": "query", "list": "search", "srsearch": title, "srlimit": n, "format": "json"}

    r = requests.get(API, params=params, headers=HEADERS, timeout=10)
    r.raise_for_status()
    data = r.json()

    if "query" not in data:
        return []
    results = data["query"]["search"]
    if not results:
        return []

    CACHE[title] = [f"https://en.wikipedia.org/wiki/{r['title'].replace(' ', '_')}" for r in results]

    with open(CACHE_FILE, "w+") as f:
        f.write(json.dumps(CACHE, indent=4))

    return CACHE[title]


def top_n_urls(query: str, n: int = 15, max_retries: int = 5) -> list[str]:
    retries = 0
    result = wiki_result(query, n)
    while len(result) == 0 and retries <= max_retries:
        time.sleep(0.1 * (retries + 1))
        result = wiki_result(query, n)
        retries += 1
    if len(result) == 0:
        print("AAAAAAAAAA (search)")

    return result[:n]
