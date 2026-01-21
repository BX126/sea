from __future__ import annotations

import requests
from urllib.parse import quote
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

@dataclass
class OrphaDisease:
    orphacode: int
    preferred_term: str
    definition: Optional[str]
    orphanet_url: Optional[str]
    synonyms: List[str]

def fetch_orphanet_short_description(
    disease_name: str,
    *,
    lang: str = "en",
    timeout_s: float = 15.0,
    session: Optional[requests.Session] = None,
):

    lang = (lang or "en").strip().lower()
    base = "https://api.orphadata.com"
    path = f"/rd-cross-referencing/orphacodes/names/{quote(disease_name.strip())}"
    url = f"{base}{path}"

    s = session or requests.Session()
    resp = s.get(url, params={"lang": lang}, timeout=timeout_s)
    if resp.status_code == 403:
        raise RuntimeError(
            "Access denied (403) from Orphadata API. "
            "If this persists, check Orphadata/Orphanet access conditions."
        )
    if resp.status_code == 404:
        return f"No Orphanet match found for: {disease_name!r}"
    resp.raise_for_status()

    payload: Dict[str, Any] = resp.json()
    results: Union[None, Dict[str, Any], List[Dict[str, Any]]] = (
        payload.get("data", {}) or {}
    ).get("results")

    # The OpenAPI schema often shows `results` as an object, but be robust anyway.
    candidates: List[Dict[str, Any]]
    if results is None:
        return f"No Orphanet match found for: {disease_name!r}"
    if isinstance(results, list):
        candidates = results
    elif isinstance(results, dict):
        candidates = [results]
    else:
        return f"Unexpected `results` type: {type(results)}"

    best = candidates[0]

    orphacode = best.get("ORPHAcode")
    preferred_term = best.get("Preferred term") or ""
    orphanet_url = best.get("OrphanetURL")
    synonyms = best.get("Synonym") or []
    if not isinstance(synonyms, list):
        synonyms = [str(synonyms)]

    # Definition is typically under SummaryInformation: [{"Definition": "..."}]
    definition: Optional[str] = None
    summary_info = best.get("SummaryInformation") or []
    if isinstance(summary_info, list):
        for item in summary_info:
            if isinstance(item, dict) and item.get("Definition"):
                definition = str(item["Definition"]).strip()
                break
    elif isinstance(summary_info, dict) and summary_info.get("Definition"):
        definition = str(summary_info["Definition"]).strip()

    if not isinstance(orphacode, int):
        # Some APIs serialize numbers as strings; try to coerce.
        try:
            orphacode = int(str(orphacode))
        except Exception as e:
            raise RuntimeError(f"Could not parse ORPHAcode: {orphacode!r}") from e

    return OrphaDisease(
        orphacode=orphacode,
        preferred_term=str(preferred_term).strip(),
        definition=definition,
        orphanet_url=str(orphanet_url).strip() if orphanet_url else None,
        synonyms=[str(x).strip() for x in synonyms if str(x).strip()],
    )

def search_by_name(disease_name: str, *, lang: str = "en") -> Optional[str]:
    disease_name = disease_name.replace("_", " ").strip()
    d = fetch_orphanet_short_description(disease_name, lang=lang)
    if type(d) == str:
        print(d)
        return None
    results = {
        "preferred_term": d.preferred_term,
        "orphacode": d.orphacode,
        "definition": d.definition,
        "orphanet_url": d.orphanet_url,
        "synonyms": d.synonyms
    }
    return results

import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

train_set = pd.read_csv("./raw_data/train.csv")
val_set = pd.read_csv("./raw_data/val.csv")

all_train_labels = train_set["final_diagnosis"].unique()
all_val_labels = val_set["final_diagnosis"].unique()

def process_label(label):
    info = search_by_name(label)
    return (label, info)

# For train set
all_train_lables_with_info = {}
with ThreadPoolExecutor(max_workers=4) as executor:
    future_to_label = {executor.submit(process_label, label): label for label in all_train_labels}
    for future in tqdm(as_completed(future_to_label), total=len(all_train_labels), desc="Extracting train labels info"):
        label, info = future.result()
        if info:
            all_train_lables_with_info[label] = info
        with open("./raw_data/all_train_labels_with_info.json", "w", encoding="utf-8") as f:
            json.dump(all_train_lables_with_info, f, ensure_ascii=False)

# For val set
all_val_lables_with_info = {}
with ThreadPoolExecutor(max_workers=4) as executor:
    future_to_label = {executor.submit(process_label, label): label for label in all_val_labels}
    for future in tqdm(as_completed(future_to_label), total=len(all_val_labels), desc="Extracting val labels info"):
        label, info = future.result()
        if info:
            all_val_lables_with_info[label] = info
        with open("./raw_data/all_val_labels_with_info.json", "w", encoding="utf-8") as f:
            json.dump(all_val_lables_with_info, f, ensure_ascii=False)

