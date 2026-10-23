"""Upload a verified bundle to an unpublished Zenodo draft; never publish.

The token is read from a local owner-only file or standard input, never
command-line text or a URL. No token is printed or stored in upload state.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import urllib.parse

import requests

from checkpoint_bundle import read_manifest, sha256_file


def token_from_file(path: Path) -> str:
    mode = path.stat()
    if mode.st_uid != os.getuid() or stat.S_IMODE(mode.st_mode) & 0o077:
        raise ValueError("Token file must be owned by you and readable only by you (chmod 600)")
    token = path.read_text().strip()
    if not token or any(ch.isspace() for ch in token):
        raise ValueError("Token file must contain one nonempty token")
    return token


def checked_json(response):
    if not response.ok:
        # Avoid dumping potentially sensitive responses or request headers.
        raise RuntimeError(f"Zenodo API returned HTTP {response.status_code}")
    return response.json()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ProgressReader:
    def __init__(self, stream, name: str, size: int):
        self.stream, self.name, self.size = stream, name, size
        self.sent = 0
        self.last_report = time.monotonic()

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def read(self, size=-1):
        chunk = self.stream.read(size)
        self.sent += len(chunk)
        if time.monotonic() - self.last_report >= 30:
            print(f"{self.name}: {100 * self.sent / max(1, self.size):.1f}% sent", flush=True)
            self.last_report = time.monotonic()
        return chunk


def upload(bundle: Path, token_file: Path, state_file: Path,
           draft_id: int | None = None, sandbox: bool = False,
           token: str | None = None, workers: int = 1):
    manifest = read_manifest(bundle)
    if len(manifest["models"]) != 21:
        raise ValueError("Expected a complete, verified 21-model bundle")
    expected = {}
    for line in (bundle / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if Path(name).name != name:
            raise ValueError("Unexpected checksum path")
        expected[name] = digest
    files = sorted(p for p in bundle.iterdir() if p.is_file())
    if set(expected) != {p.name for p in files if p.name != "SHA256SUMS"}:
        raise ValueError("Staged files differ from SHA256SUMS")
    if len(files) > 100 or sum(p.stat().st_size for p in files) > 50_000_000_000:
        raise ValueError("Bundle exceeds the standard Zenodo quota")
    for path in files:
        if path.name in expected and sha256_file(path) != expected[path.name]:
            raise ValueError(f"Staged file checksum mismatch: {path.name}")
    origin = "https://sandbox.zenodo.org" if sandbox else "https://zenodo.org"
    session = requests.Session()
    credential = token if token is not None else token_from_file(token_file)
    if not credential or any(ch.isspace() for ch in credential):
        raise ValueError("Expected one nonempty token")
    session.headers.update({"Authorization": f"Bearer {credential}"})
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if state["origin"] != origin:
            raise ValueError("Saved upload belongs to a different Zenodo instance")
        if draft_id is not None and draft_id != state["draft_id"]:
            raise ValueError("Draft ID differs from saved upload state")
        draft_id = state["draft_id"]
    if draft_id is None:
        draft = checked_json(session.post(f"{origin}/api/deposit/depositions", json={}, timeout=60))
        draft_id = draft["id"]
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps({"origin": origin, "draft_id": draft_id}, indent=2) + "\n")
    else:
        draft = checked_json(session.get(f"{origin}/api/deposit/depositions/{draft_id}", timeout=60))
    if draft.get("submitted") or draft.get("state") != "unsubmitted":
        raise ValueError("Refusing to modify a published or submitted record")
    bucket = draft["links"]["bucket"]
    parsed = urllib.parse.urlsplit(bucket)
    if parsed.scheme != "https" or parsed.netloc != urllib.parse.urlsplit(origin).netloc:
        raise ValueError("Unexpected upload bucket host")
    remote = {item.get("filename", item.get("name")): item for item in draft.get("files", [])}
    print(f"Unpublished draft: {origin}/uploads/{draft_id}", flush=True)
    print(f"Draft currently contains {len(remote)} completed files", flush=True)
    def upload_one(path):
        checksum = md5_file(path)
        if path.name in remote:
            old = remote[path.name]
            if old.get("checksum", "").removeprefix("md5:") == checksum:
                print(f"Already uploaded: {path.name}", flush=True)
                return path.name, checksum
            raise ValueError(f"Remote filename exists with different contents: {path.name}")
        worker_session = requests.Session()
        worker_session.headers.update(session.headers)
        print(f"Uploading {path.name} ({path.stat().st_size / 1e9:.3f} GB)", flush=True)
        for attempt in range(4):
            with path.open("rb") as stream:
                response = worker_session.put(
                    f"{bucket}/{urllib.parse.quote(path.name, safe='')}",
                    data=ProgressReader(stream, path.name, path.stat().st_size),
                    headers={"Content-Type": "application/octet-stream"}, timeout=(60, 1800))
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                print(f"Retrying {path.name} after HTTP {response.status_code}", flush=True)
                time.sleep(min(10 * (attempt + 1), 30))
                continue
            if not response.ok:
                print(f"Upload failed: {path.name}, HTTP {response.status_code}", flush=True)
            result = checked_json(response)
            break
        if result.get("checksum", "").removeprefix("md5:") != checksum:
            raise ValueError(f"Remote checksum differs: {path.name}")
        print(f"Verified upload: {path.name}", flush=True)
        return path.name, checksum
    if workers == 1:
        uploaded = dict(upload_one(path) for path in files)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [executor.submit(upload_one, path) for path in files]
        try:
            uploaded = dict(f.result() for f in as_completed(futures))
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    draft = checked_json(session.get(f"{origin}/api/deposit/depositions/{draft_id}", timeout=60))
    if draft.get("submitted"):
        raise RuntimeError("Unexpected published state")
    final_files = {r.get("filename", r.get("name")): r for r in draft.get("files", [])}
    for name, expected_md5 in uploaded.items():
        if name not in final_files or final_files[name].get("checksum", "").removeprefix("md5:") != expected_md5:
            raise ValueError(f"Final draft file verification failed: {name}")
    print(f"Upload verified. Draft remains unpublished: {origin}/uploads/{draft_id}", flush=True)
    print("Creators, licensing, metadata and release still require your review.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    credentials = parser.add_mutually_exclusive_group(required=True)
    credentials.add_argument("--token-file", type=Path)
    credentials.add_argument("--token-stdin", action="store_true",
                             help="Read one token line from standard input without saving it")
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--draft-id", type=int)
    parser.add_argument("--sandbox", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=1)
    args = parser.parse_args()
    token = sys.stdin.readline().strip() if args.token_stdin else None
    upload(args.bundle, args.token_file, args.state_file, args.draft_id, args.sandbox, token, args.workers)


if __name__ == "__main__":
    main()
