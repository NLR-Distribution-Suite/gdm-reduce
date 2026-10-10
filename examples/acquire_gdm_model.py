"""Acquire a published GDM model and save a portable local copy."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from gdm.systems.distribution import DistributionSystem
from gdmloader.constants import GCS_CASE_SOURCE
from gdmloader.source import SystemLoader
from prepare_circuit_models import save_portable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="p5r")
    parser.add_argument(
        "--output", type=Path, default=Path("results/acquired/p5r.json")
    )
    arguments = parser.parse_args()
    loader = SystemLoader()
    loader.add_source(GCS_CASE_SOURCE)
    system = loader.load_dataset(
        system_type=DistributionSystem,
        source_name=GCS_CASE_SOURCE.name,
        dataset_name=arguments.dataset,
    )
    removed = save_portable(system, arguments.output)
    provenance = {
        "dataset": arguments.dataset,
        "source": GCS_CASE_SOURCE.name,
        "gdmloader_version": version("gdmloader"),
        "gdm_version": version("grid-data-models"),
        "saved_json_sha256": hashlib.sha256(arguments.output.read_bytes()).hexdigest(),
        "removed_converter_annotations": removed,
    }
    arguments.output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"Acquired {arguments.dataset}: {arguments.output}")


if __name__ == "__main__":
    main()
