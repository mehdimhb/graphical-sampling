"""Read-only consistency checks for the final audit; no sampling or DI imports."""
from pathlib import Path
from collections import Counter
from decimal import Decimal
import csv
import hashlib
import json
import re

OUT = Path(__file__).resolve().parent

def read_csv(path):
    with (OUT / path).open(newline='') as handle:
        return list(csv.DictReader(handle))

def read_json(path):
    return json.loads((OUT / path).read_text())

rep = read_csv('evidence/replications/per_setting_replication_ledger.csv')
match = read_csv('evidence/matching/row_provenance.csv')
key = lambda row: tuple(row[field] for field in ('Population','Probability','Method','n','N'))
assert len(rep) == len(match) == 162
assert len({key(row) for row in rep}) == 162
assert {key(row) for row in rep} == {key(row) for row in match}
assert Counter(row['status'] for row in match) == {
    'saved_MC_exact':104, 'saved_MC_DI_mean_discrepancy':2,
    'MC_retained_output_missing':32, 'saved_design_exact_import_unresolved':24}
for row in match:
    ledger = next(r for r in rep if key(r) == key(row))
    assert Decimal(row['legacy_Dm']) == Decimal(ledger['Dm'])
    assert Decimal(row['legacy_Ds']) == Decimal(ledger['Ds'])
checks = read_json('evidence/replications/replication_checks.json')
assert checks['finite_enumeration_rows'] == 21368
assert checks['all_finite_enumeration_rows_valid_positive']
assert checks['effective_MC_counts_independently_logged'] == 0
provenance = read_json('evidence/matching/provenance_checks.json')
assert all(row['exact_mean_SD_rows'] == (138 if 'Monte Carlo' in row['name'] else 162)
           for row in provenance['CSV_checks'])
assert all(row['current_byte_identical'] for row in provenance['figure_continuity'])
assert provenance['June5_OT_test']['exact_mean_SD_matches_to_legacy'] == 0

kernel = read_csv('evidence/kernel/kernel_history.csv')
assert len(kernel) == 6
assert all(row['kernel'] == 'tophat' and row['bandwidth'] == 'scott' for row in kernel)
route = read_json('evidence/kernel/historical_execution_path.json')
assert not route['KDE_override_present'] and not route['DI_formula_override_present']
assert route['runtime_import_version_frozen'] is False

counter = read_json('assignment_counterexample.json')
assert counter['all_checks_passed']
assert counter['euclidean']['assignment_before'] == [0,1]
assert counter['euclidean']['assignment_after'] == [1,0]
assert counter['sqeuclidean']['assignment_after'] == [0,1]
assert Decimal(counter['high_precision_after_gap_identity_minus_swap']) > Decimal('0.1129')
for name in ('identity','swap'):
    costs=counter['sqeuclidean']
    assert costs[name+'_cost_after'] - costs[name+'_cost_before'] == 12

proposal = read_json('proposed_passages.json')
source = Path(proposal['source'])
assert hashlib.sha256(source.read_bytes()).hexdigest() == proposal['source_sha256']
lines = source.read_text().splitlines(keepends=True)
assert len(proposal['passages']) == 15
assert len({p['id'] for p in proposal['passages']}) == 15
for p in proposal['passages']:
    assert ''.join(lines[p['start_line']-1:p['end_line']]) == p['exact_existing_latex'], p['id']
assert proposal['originals_modified'] is False
assert next(p for p in proposal['passages'] if p['id']=='section4_replications')['decision']=='Both A and B'

# Check deliverable relative links without following arbitrary external links.
linked=[]
for name in ('REPORT.md','PROPOSED_CORRECTIONS.md'):
    for target in re.findall(r'\]\(([^)]+)\)', (OUT/name).read_text()):
        if not target.startswith(('https:','http:','#','/')):
            assert (OUT/target.split('#')[0]).exists(), (name,target)
            linked.append(target)

result=dict(passed=True,setting_rows=162,MC_exact_links=104,
            MC_discrepant=2,MC_unlogged=32,finite_designs=24,
            finite_enumeration_entries=21368,kernel_source_snapshots=6,
            verified_exact_quotation_fragments=15,relative_links_checked=len(linked),
            new_Monte_Carlo_samples=0,DI_scores_recomputed=0,
            scope='Evidence consistency plus deterministic two-pair assignment; historical runtime uncertainty remains explicit.')
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
