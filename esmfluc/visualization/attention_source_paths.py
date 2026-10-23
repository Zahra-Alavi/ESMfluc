"""Path resolution shared by retained attention-source visualizers."""

from pathlib import Path


def resolve_existing_path(path_value, result_root, pipeline_dir):
    path = Path(str(path_value))
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([
            result_root / path,
            result_root.parent / path,
            pipeline_dir / path,
            Path.cwd() / path,
            path,
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_manifest_path(result_root, manifest_tsv=None):
    path = Path(manifest_tsv).expanduser() if manifest_tsv else result_root / "manifest.tsv"
    if not path.is_absolute():
        path = result_root / path
    return path.resolve()
