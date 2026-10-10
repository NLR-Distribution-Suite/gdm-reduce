"""Copy DiTTo's tested OpenDSS circuits into standalone GDM fixtures."""

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import DistributionBus
from infrasys import SupplementalAttribute


def save_portable(system: DistributionSystem, output: Path) -> list[str]:
    removed = []
    for annotation in list(system.get_supplemental_attributes(SupplementalAttribute)):
        if type(annotation).__module__.startswith("ditto."):
            removed.append(type(annotation).__name__)
            system.remove_supplemental_attribute(annotation)
    output.parent.mkdir(parents=True, exist_ok=True)
    system.to_json(output, overwrite=True)
    restored = DistributionSystem.from_json(output)
    assert {bus.uuid for bus in system.get_components(DistributionBus)} == {
        bus.uuid for bus in restored.get_components(DistributionBus)
    }
    return sorted(removed)


def convert_models(source: Path, destination: Path, cases: list[str]) -> list[dict]:
    from ditto.readers.opendss.reader import Reader

    records = []
    for name in cases:
        circuit = source / name / "Master.dss"
        previous_directory = Path.cwd()
        try:
            system = Reader(circuit).get_system()
        finally:
            os.chdir(previous_directory)
        removed = save_portable(system, destination / name / "model.json")
        checksums = {
            str(path.relative_to(circuit.parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(circuit.parent.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".dss", ".csv"}
        }
        records.append(
            {
                "name": name,
                "source": f"ditto/tests/data/opendss_circuit_models/{name}/Master.dss",
                "source_sha256": checksums,
                "model": f"{name}/model.json",
                "ditto_version": version("NREL-ditto"),
                "gdm_version": version("grid-data-models"),
                "buses": len(list(system.get_components(DistributionBus))),
                "removed_converter_annotations": removed,
                "electrical_normalization": "none",
            }
        )
        print(f"Converted {name}: {records[-1]['buses']} buses", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tests/data/circuits"))
    parser.add_argument(
        "--cases", nargs="+", default=["ieee13", "P4U", "ckt7", "ckt24", "SFO"]
    )
    arguments = parser.parse_args()
    convert_models(
        arguments.source.resolve(), arguments.output.resolve(), arguments.cases
    )


if __name__ == "__main__":
    main()
