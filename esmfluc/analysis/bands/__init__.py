"""Reproducible signed-contribution band analyses for ESMfluc.

The package intentionally avoids importing analysis modules at package import
time. Most commands have optional scientific dependencies and operate on large
result files, so callers should import the specific module they need.
"""

__all__ = ()
