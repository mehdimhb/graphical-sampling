#!/usr/bin/env bash
# Read only original PDFs and archive members; write only into this audit directory.
set -euo pipefail
AUDIT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname -- "$AUDIT_ROOT")"
python3 - "$AUDIT_ROOT" <<'PY'
from pathlib import Path
import sys,zipfile,json,hashlib
h=Path(sys.argv[1]);zpath=Path('/home/bardia/Downloads/Spread_Measure_Submission_FINAL.zip')
with zipfile.ZipFile(zpath) as z:
    member='Spread_Measure_Submission_FINAL/figures/simulated_result_measure_medoid.pdf'
    (h/'evidence/submission_simulated_result_measure_medoid.pdf').write_bytes(z.read(member))
    member='Spread_Measure_Submission_FINAL/Measure_blinded_clean.tex'
    (h/'evidence/submission_measure.tex').write_bytes(z.read(member))
PY
pdftotext -layout '/home/bardia/Downloads/JSSAM-2025-203.R1_Proof_hi_0.pdf' "$AUDIT_ROOT/evidence/JSSAM_R1_proof.txt"
pdfseparate -f 31 -l 31 '/home/bardia/Downloads/JSSAM-2025-203.R1_Proof_hi_0.pdf' "$AUDIT_ROOT/evidence/proof-page-%d.pdf"
pdftocairo -svg "$PROJECT_ROOT/simulations/plots/simulated_result_measure_medoid.pdf" "$AUDIT_ROOT/evidence/original_synthetic.svg"
pdftocairo -svg "$AUDIT_ROOT/evidence/submission_simulated_result_measure_medoid.pdf" "$AUDIT_ROOT/evidence/submission_synthetic.svg"
pdftocairo -svg "$AUDIT_ROOT/evidence/proof-page-31.pdf" "$AUDIT_ROOT/evidence/proof_synthetic.svg"
pdftotext -bbox "$PROJECT_ROOT/simulations/plots/simulated_result_measure_medoid.pdf" "$AUDIT_ROOT/evidence/original_synthetic_bbox.html"
pdftotext -bbox "$AUDIT_ROOT/evidence/submission_simulated_result_measure_medoid.pdf" "$AUDIT_ROOT/evidence/submission_synthetic_bbox.html"
pdftotext -bbox "$AUDIT_ROOT/evidence/proof-page-31.pdf" "$AUDIT_ROOT/evidence/proof_synthetic_bbox.html"
