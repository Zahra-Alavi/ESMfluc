"""DynaMine API client for the mdCATH benchmark."""
from pathlib import Path
import random
import string
import time
import sys
import numpy as np
import pandas as pd
import requests
_B2B_API_BASE = "https://bio2byte.be/msatools/api/"
_B2B_BATCH_MAX = 50

def _random_token(length: int = 10) -> str:
    """Generate a random alphanumeric session token for the Bio2Byte API."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

def parse_fasta(path: str | Path) -> dict[str, str]:
    records: dict[str, str] = {}
    with open(path) as fh:
        name, buf = None, []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    if name in records:
                        raise ValueError(f"Duplicate FASTA name: {name}")
                    records[name] = "".join(buf)
                name = line[1:].strip()
                if not name:
                    raise ValueError("Empty FASTA header")
                buf = []
            else:
                buf.append(line)
        if name:
            if name in records:
                raise ValueError(f"Duplicate FASTA name: {name}")
            records[name] = "".join(buf)
    if not records:
        raise ValueError(f"No FASTA records in {path}")
    return records

def _b2b_submit(session, sequences: dict[str, str], token: str) -> str:
    """POST one batch. Returns hash_id."""
    payload: dict = {"tool_list": ["dynamine"], "token": token}
    payload.update(sequences)
    resp = session.post(_B2B_API_BASE, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    hash_id = data.get("hash_id")
    if not hash_id:
        raise RuntimeError(f"API did not return hash_id: {data}")
    return str(hash_id)

def _b2b_poll(session, hash_id: str, poll_interval: int, max_polls: int) -> list[dict]:
    """Poll queue until complete, then retrieve and return results list."""
    queue_url  = f"{_B2B_API_BASE}queue/{hash_id}/"
    result_url = f"{_B2B_API_BASE}{hash_id}/"

    for attempt in range(max_polls):
        resp = session.get(queue_url, timeout=30, allow_redirects=False)
        # 303 = explicit redirect to results.
        # 200 = results ready (observed in practice; status field may be 200).
        # 202 = still queued/processing.
        if resp.status_code == 303:
            break
        try:
            data = resp.json()
        except Exception:
            time.sleep(poll_interval)
            continue
        if data.get("status") == 500 or data.get("Failure"):
            raise RuntimeError(f"Server-side prediction failure: {data.get('Failure')}")
        http_status = data.get("status", resp.status_code)
        # 202 means still processing; anything else (200, 303) means done.
        if http_status != 202:
            break
        if attempt % 3 == 0:
            remaining = data.get("request_text", "still processing")
            print(f"      poll {attempt + 1}/{max_polls}: {remaining}")
        time.sleep(poll_interval)
    else:
        raise TimeoutError(
            f"DynaMine API timed out after {max_polls} polls "
            f"({max_polls * poll_interval}s)"
        )

    resp = session.get(result_url, timeout=60)
    resp.raise_for_status()
    return resp.json().get("results", [])

def run_dynamine_api(
    fasta_records: dict[str, str],
    token: str,
    batch_size: int = 40,
    poll_interval: int = 20,
    max_polls: int = 90,
) -> pd.DataFrame:
    """
    Submit test sequences to the Bio2Byte DynaMine API in batches.
    Returns long DataFrame: name, res_idx (1-based), dynamine_bb.
    Lower DynaMine backbone S² predictions indicate greater dynamics. Raw
    regression output is retained even when it falls slightly outside [0, 1].
    """
    batch_size = min(batch_size, _B2B_BATCH_MAX)
    names = list(fasta_records.keys())
    batches = [names[i: i + batch_size] for i in range(0, len(names), batch_size)]

    session = requests.Session()
    all_rows: list[dict] = []

    for b_idx, batch_names in enumerate(batches, 1):
        batch_seqs = {n: fasta_records[n] for n in batch_names}
        print(f"    Batch {b_idx}/{len(batches)}: {len(batch_seqs)} sequences …")
        try:
            hash_id = _b2b_submit(session, batch_seqs, token)
            results = _b2b_poll(session, hash_id, poll_interval, max_polls)
        except Exception as exc:
            print(f"      [WARN] Batch {b_idx} failed: {exc}")
            for n in batch_names:
                for j in range(len(fasta_records[n])):
                    all_rows.append({"name": n, "res_idx": j + 1, "dynamine_bb": np.nan})
            continue

        # Index returned results by proteinID.
        # The API returns per-predictor keys; DynaMine backbone S² is under
        # "backbone", not "dynamine" ("dynamine" key does not exist).
        result_index: dict[str, list] = {
            r["proteinID"]: r.get("backbone", []) for r in results
        }

        for name in batch_names:
            preds = result_index.get(name, [])
            L_expected = len(fasta_records[name])
            if not preds:
                print(f"      [WARN] No predictions for {name}")
                for j in range(L_expected):
                    all_rows.append({"name": name, "res_idx": j + 1, "dynamine_bb": np.nan})
                continue
            for j, val in enumerate(preds[:L_expected]):
                all_rows.append({
                    "name": name,
                    "res_idx": j + 1,
                    "dynamine_bb": float(val) if val is not None else np.nan,
                })

    return pd.DataFrame(all_rows)
