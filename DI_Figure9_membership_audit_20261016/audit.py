"""Audit saved Figure 9 geometry. No DI, clustering or reference search is run.

Only this new audit directory is written. Existing scientific files are read.
Run from the project root with .venv/bin/python and single-thread environment.
"""
from pathlib import Path
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import csv
import hashlib
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from fractions import Fraction

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "simulations/temporary_random2D_DI_two_six_panel_figures.py"
NOTEBOOK = ROOT / "simulations/jupyters/13_2D_grid_DI_translation_rotation.ipynb"
PDF = ROOT / "simulations/results/temporary_2D_relative_rotation/grid_20x20_N400_DI_original_shift_rotation90_rotation45_large_2x2_n20.pdf"
TOL = 1e-12  # Scientific membership tolerance, never a fraction of grid spacing.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform_xml(element, points):
    a, b, c, d, tx, ty = map(float, element.get("transform").split())
    return np.asarray(points) @ np.array([[a, b], [c, d]]) + [tx, ty]


def recover_pdf_samples():
    """Recover marker positions from original vector PDF, not from a new run.

    MuPDF trace includes the marker transforms and rectangular axis spines.
    PDF float precision is checked separately; exact grid coordinates are then
    identified. The looser PDF decoding bound is NOT used to test rotations.
    """
    with tempfile.TemporaryDirectory(prefix="figure9_audit_") as tmp:
        trace = Path(tmp) / "trace.xml"
        subprocess.run(["mutool", "draw", "-F", "trace", "-o", str(trace), str(PDF)],
                       check=True, capture_output=True)
        root = ET.parse(trace).getroot()
    vertical = []
    for element in root.iter("stroke_path"):
        if element.get("linewidth") == ".8" and element.get("linecap") == "2,2,2":
            points = [(float(e.get("x")), float(e.get("y"))) for e in element
                      if e.tag in ("moveto", "lineto")]
            if len(points) == 2:
                q = transform_xml(element, points)
                if q[0, 0] == q[1, 0] and abs(q[0, 1] - q[1, 1]) > 300:
                    vertical.append(q)
    xs = sorted(set(q[0, 0] for q in vertical))
    ys = sorted(set(y for q in vertical for x, y in q))
    assert len(xs) == len(ys) == 4
    red = []
    for element in root.iter("fill_path"):
        color = element.get("color", "").split()
        if (len(color) == 3 and abs(float(color[0]) - 229/255) < 1e-6
                and abs(float(color[1]) - 57/255) < 1e-6
                and abs(float(color[2]) - 53/255) < 1e-6):
            red.append(transform_xml(element, [[0, 0]])[0])
    red = np.asarray(red)
    assert len(red) == 81  # 80 panel points and one legend marker.
    rectangles = [(xs[0], xs[1], ys[0], ys[1]),
                  (xs[2], xs[3], ys[0], ys[1]),
                  (xs[0], xs[1], ys[2], ys[3]),
                  (xs[2], xs[3], ys[2], ys[3])]
    samples, residuals = {}, {}
    for panel, (x0, x1, y0, y1) in zip("abcd", rectangles):
        markers = red[(red[:, 0] >= x0-.001) & (red[:, 0] <= x1+.001)
                      & (red[:, 1] >= y0-.001) & (red[:, 1] <= y1+.001)]
        assert len(markers) == 20
        normalized = (markers - [x0, y1]) / [x1-x0, y0-y1]
        grid = np.rint(normalized * 19).astype(int)
        residuals[panel] = float(np.max(np.abs(normalized - grid/19)))
        assert residuals[panel] < 2e-6  # Vector-PDF extraction precision only.
        assert np.all((grid >= 0) & (grid <= 19))
        assert len(set(map(tuple, grid))) == 20
        samples[panel] = grid
    return samples, residuals


def main():
    # Freeze relevant originals; only new deliverables under OUT are written.
    protected = [SOURCE, NOTEBOOK, PDF,
        ROOT / "DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex",
        ROOT / "DI_historical_implementation_audit_20261016/evidence/submission_measure.tex"]
    before = {str(p): sha(p) for p in protected}
    git_blob_checks = {}
    for p in (SOURCE, NOTEBOOK):
        blob = subprocess.check_output(["git", "-C", str(ROOT), "show",
                                        "046dbca43:" + str(p.relative_to(ROOT))])
        assert blob == p.read_bytes()
        git_blob_checks[str(p.relative_to(ROOT))] = True
    nb = json.loads(NOTEBOOK.read_text())
    assert "".join(nb["cells"][1]["source"]) == SOURCE.read_text()
    archive_matches = []
    for archive in sorted(Path("/home/bardia/Downloads").glob("*.zip")):
        try:
            with zipfile.ZipFile(archive) as z:
                for name in z.namelist():
                    if name.endswith("/figures/DI_rotation_4x5.pdf") or name == "figures/DI_rotation_4x5.pdf":
                        digest = hashlib.sha256(z.read(name)).hexdigest()
                        assert digest == sha(PDF)
                        archive_matches.append(dict(archive=str(archive), member=name, sha256=digest))
        except zipfile.BadZipFile:
            continue
    assert len(archive_matches) >= 1
    samples, residuals = recover_pdf_samples()
    original = samples["a"] / 19
    # Independent, compact description of the complete original sample.
    expected = np.array([(i, j) for j in (2, 7, 12, 17) for i in (1, 5, 9, 13, 17)])
    assert np.array_equal(samples["a"], expected)
    population_grid = np.array([(i, j) for j in range(20) for i in range(20)])
    population = population_grid / 19
    sample_center = original.mean(axis=0)
    assert np.allclose(sample_center, [9/19, 9.5/19], atol=TOL, rtol=0)
    scenarios = [
        ("a", 0, sample_center, [0., 0.], "a"),
        ("b", 0, sample_center, [1/19, -1/19], "b"),
        ("c", 90, sample_center, [0., 0.], "c"),
        ("d", 45, sample_center, [0., 0.], "d"),
        ("population_center_90", 90, np.array([.5, .5]), [0., 0.], None),
        ("population_center_45", 45, np.array([.5, .5]), [0., 0.], None),
    ]
    summary, details = [], []
    for name, angle, center, shift, panel in scenarios:
        a = np.deg2rad(angle)
        rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        targets = center + (original-center) @ rotation.T + shift
        distances = cdist(targets, population)
        nearest = distances.min(axis=1)
        is_unit = nearest <= TOL
        inside = np.all((targets >= -TOL) & (targets <= 1+TOL), axis=1)
        row = dict(case=name, angle_degrees=angle, center_x=float(center[0]),
                   center_y=float(center[1]), shift_x=shift[0], shift_y=shift[1],
                   exact_units=int(is_unit.sum()), inside_off_grid=int((inside & ~is_unit).sum()),
                   outside=int((~inside).sum()), inside_total=int(inside.sum()),
                   displayed_units=20 if panel else "", maximum_projection_distance="",
                   observed_assignment_is_optimal="")
        if panel:
            observed = samples[panel]/19
            rows, cols = linear_sum_assignment(distances)
            actual_cost = np.linalg.norm(observed-targets, axis=1).sum()
            optimal_cost = distances[rows, cols].sum()
            assert abs(actual_cost-optimal_cost) < TOL
            assert np.array_equal(samples[panel], population_grid[cols])
            row["observed_assignment_is_optimal"] = True
            row["maximum_projection_distance"] = float(np.linalg.norm(observed-targets, axis=1).max())
        else:
            observed = None
        summary.append(row)
        for k, target in enumerate(targets):
            details.append(dict(case=name, point=k, original_i=int(expected[k, 0]),
                original_j=int(expected[k, 1]), target_x=float(target[0]), target_y=float(target[1]),
                exact_population_unit=bool(is_unit[k]), inside_boundary=bool(inside[k]),
                nearest_population_distance=float(nearest[k]),
                displayed_i=int(samples[panel][k, 0]) if panel else "",
                displayed_j=int(samples[panel][k, 1]) if panel else ""))
    # Independent exact arithmetic for the two 90-degree rotations.
    exact90 = [(19-j, i) for i, j in map(tuple, expected)]
    assert all(0 <= i <= 19 and 0 <= j <= 19 for i, j in exact90)
    around_sample90 = [(Fraction(37, 2)-int(j), Fraction(1, 2)+int(i)) for i, j in expected]
    assert all(x.denominator == y.denominator == 2 for x, y in around_sample90)
    # Actual c is also a rigid 90-degree rotation, but about (9,10)/19.
    assert np.array_equal(samples["c"], np.array([(19-j, i+1) for i, j in expected]))
    assert np.array_equal(samples["b"], expected + [1, -1])
    # The d panel is not any rigid rotation: ordered pairwise distances change.
    d_distance_change = float(np.max(np.abs(cdist(original, original)-cdist(samples["d"]/19, samples["d"]/19))))
    assert d_distance_change > .01
    for filename, records in (("panel_and_rotation_checks.csv", summary), ("point_checks.csv", details)):
        with (OUT/filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    with (OUT/"valid_centered_90_sample.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["point", "original_i", "original_j", "rotated_i", "rotated_j", "rotated_x", "rotated_y", "zero_based_population_index"])
        for k, ((i,j), (ri,rj)) in enumerate(zip(expected, exact90)):
            writer.writerow([k, i, j, ri, rj, ri/19, rj/19, 20*rj+ri])
    after = {str(p): sha(p) for p in protected}
    assert before == after
    result = dict(all_checks_passed=True, DI_runs=0, optimization_runs=0,
        tolerance=TOL, population="{(i/19,j/19):i,j=0,...,19}", N=400, n=20,
        original_sample_integer_grid=expected.tolist(),
        sample_center=[9/19, .5], population_center=[.5,.5],
        pdf_grid_decoding_residuals=residuals, pdf_decoding_tolerance=2e-6,
        pdf_decoding_note="PDF float/rendering tolerance is separate from exact coordinate membership tests at 1e-12.",
        source_historical_commit="046dbca43", source_unchanged_from_commit=git_blob_checks,
        notebook_source_equals_script=True, matching_manuscript_archives=archive_matches,
        panel_checks=summary, maximum_d_pairwise_distance_change=d_distance_change,
        original_protected_sha256=after,
        scope="September final archives/current manuscript snapshot referencing DI_rotation_4x5.pdf; no claim about a different Figure 9 in an earlier proof.")
    (OUT/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    (OUT/"figure_text.txt").write_bytes(subprocess.check_output(["pdftotext", "-layout", str(PDF), "-"]))
    print(json.dumps(dict(all_checks_passed=True, archive_matches=len(archive_matches),
                         panel_checks=summary, protected_unchanged=before == after), indent=2))


if __name__ == "__main__":
    main()
