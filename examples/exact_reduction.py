"""Run an exact loaded-bus reduction with an independent full-network solve."""

import numpy as np

from gdm_reduce import ElectricalNetwork, kron_reduce, solve_voltages


def main() -> None:
    network = ElectricalNetwork(
        admittance=np.array([[10, -10, 0], [-10, 20, -10], [0, -10, 10]]),
        labels=(("source", "A"), ("middle", "A"), ("load", "A")),
        source_voltages={("source", "A"): 120},
    )
    currents = np.array([0, -1, -2], dtype=complex)
    result = kron_reduce(network, {"source", "load"})
    retained_voltage = result.solve(currents)
    reconstructed = result.reconstruct(currents, retained_voltage)
    reference = solve_voltages(network, currents)
    error = float(np.max(np.abs(reconstructed - reference)))
    np.testing.assert_allclose(reconstructed, reference, rtol=1e-10)
    print(f"Original terminals: {len(network.labels)}")
    print(f"Retained terminals: {len(result.retained_labels)}")
    print(f"Retained voltage (V): {retained_voltage}")
    print(f"Reconstructed voltage (V): {reconstructed}")
    print(f"Maximum reconstruction error (V): {error:.3g}")
    print("Exact prescribed-current equivalent; no device aggregation performed.")


if __name__ == "__main__":
    main()
