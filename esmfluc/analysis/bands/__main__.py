"""Command dispatcher for the ESMfluc signed-band analysis package."""

from __future__ import annotations

import importlib
import sys


COMMANDS = {
    "add-seed-average": "add_seed_averaged_influence",
    "extract-bands": "extract_signed_contribution_bands",
    "build-uniform-control": "build_uniform_attention_control_profiles",
    "compare-uniform-bands": "compare_observed_uniform_bands",
    "apex-component-localization": "analyze_apex_component_localization",
    "seed-reproducibility": "analyze_signed_band_seed_reproducibility",
    "annotate-biophysics": "annotate_signed_bands_with_netsurfp",
    "biophysical-enrichment": "analyze_signed_band_biophysical_enrichment",
    "apex-structure": "analyze_signed_band_apex_structure",
    "band-interval-structure": "analyze_signed_band_interval_structure",
    "model-mechanism": "analyze_signed_band_model_mechanism",
    "query-receivers": "analyze_signed_band_query_receivers",
    "build-query-structure": "build_band_query_structural_water_features",
    "audit-query-receivers": "audit_query_receivers",
    "audit-biophysics": "audit_signed_band_biophysical_pipeline",
    "plot-local-properties": "plot_local_property_results",
}


def usage() -> str:
    commands = "\n".join(f"  {name}" for name in COMMANDS)
    return (
        "usage: python -m esmfluc.analysis.bands <command> [arguments]\n\n"
        "commands:\n"
        f"{commands}\n\n"
        "Run a command with --help to see its analysis-specific arguments."
    )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(usage())
        return

    command = sys.argv.pop(1)
    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"error: unknown command {command!r}\n", file=sys.stderr)
        print(usage(), file=sys.stderr)
        raise SystemExit(2)

    sys.argv[0] = f"python -m esmfluc.analysis.bands {command}"
    module = importlib.import_module(f"{__package__}.{module_name}")
    module.main()


if __name__ == "__main__":
    main()
