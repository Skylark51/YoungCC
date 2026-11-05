"""Cross-check the workflow parser against immutable Lion Gaussian logs."""

from pathlib import Path

import pytest

from workflows.gaussian import GaussianLog, distance

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "gaussian16" / "lion-test"
ATOMS = (6, 1, 1, 1, 6, 8, 6, 6, 8, 6, 1, 1, 1, 1, 1)


def read(suffix):
    return GaussianLog((ROOT / f"MalonicAcid-scan{suffix}.out").read_text(errors="replace"), ATOMS)


def test_lion_scan_peak_and_ts_frequencies():
    scan = read("")
    assert not scan.normal  # Lion stopped the scan after bracketing the peak.
    assert len(scan.points) == 5
    assert scan.peak()[0] == pytest.approx(-345.555116933, rel=0, abs=1e-9)
    for suffix, imaginary in [("-freq", -870.1681), ("-freq-ts-freq", -1015.0054)]:
        log = read(suffix)
        assert log.normal
        frequencies = log.frequencies(log.tables[-1][2])
        assert len(frequencies) == 39
        assert [value for value in frequencies if value < 0] == [imaginary]


def test_lion_irc_endpoints_and_optimized_minima():
    endpoints = {}
    for direction, suffix in [("forward", "fwd"), ("reverse", "rev")]:
        stem = f"-freq-ts-freq-irc{suffix}"
        irc, opt, freq = (read(stem + ending) for ending in ("", "-opt", "-opt-freq"))
        assert irc.normal and opt.normal and freq.normal
        raw = irc.irc_endpoint(direction)
        assert raw.xyz[10, 0] == pytest.approx(-0.266912 if suffix == "fwd" else 0.269140)
        with pytest.raises(ValueError, match="did not complete"):
            irc.irc_endpoint("reverse" if direction == "forward" else "forward")
        geometry = opt.points[-1][1]
        reference = freq.tables[-1][2]
        assert min(freq.frequencies(reference)) > 0
        rmsd, pair = distance(geometry, reference)
        assert rmsd < 0.15 and pair < 0.35
        endpoints[direction] = geometry
    rmsd, pair = distance(endpoints["forward"], endpoints["reverse"])
    assert rmsd > 0.15 or pair > 0.35
