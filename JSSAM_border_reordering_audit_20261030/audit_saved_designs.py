"""Read historical designs; write only new audit artifacts beside this script.

No optimization, resaving, replacement designs or historical output changes.
In-memory order replays diagnose reconstruction behavior, not provenance.
Run from the repository root with its Python environment.
"""
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from graphical_sampling import Design
from graphical_sampling.index.moran import Moran
from graphical_sampling.index.voronoi import Voronoi

OUT = Path(__file__).resolve().parent


def aggregate(rows):
    parts = defaultdict(list)
    for ids, p in rows:
        p = float(p)
        if not math.isfinite(p) or p < 0:
            raise ValueError(f"Invalid retained probability fragment: {p!r}")
        parts[tuple(sorted(map(int, ids)))].append(p)
    return {s: math.fsum(p) for s, p in parts.items() if math.fsum(p) > 0}


def raw_law(d):
    return aggregate((s.ids, s.prob) for h in d._heaps for s in h)


def summarize(law, pop):
    fip = np.zeros(pop.N)
    for s, p in law.items():
        fip[list(s)] += p
    return dict(support=len(law), probability_sum=math.fsum(law.values()),
                sizes=sorted(set(map(len, law))), min_probability=min(law.values()),
                max_fip_error=float(np.max(np.abs(fip-pop.inclusions))),
                valid_ids=all(len(set(s)) == len(s) and all(0 <= i < pop.N for i in s)
                              for s in law))


def distance(a, b):
    return math.fsum(abs(a.get(s, 0)-b.get(s, 0)) for s in a.keys() | b.keys())


def membership(order):
    if not order.clusters:
        return None
    m = np.zeros((order.pop.N, len(order.clusters)))
    for j, c in enumerate(order.clusters):
        for z in c.zones:
            if len(z._shares) != len(z._indices):
                return None  # Retained flattened orders sometimes omit shares.
            np.add.at(m[:, j], z.indices, z.shares)
        for b in [c.floor, c.ceil]:
            if b is not None and b.index is not None and b.index >= 0:
                m[b.index, j] += float(b.percentage)
    return m


def cache_mean(d, name):
    value = getattr(d, name, None)
    return None if value is None else float(value[0])


def unmatchable_mass(law, m):
    # Necessary-and-sufficient eligibility for assigning one selected distinct
    # unit to each original soft cluster; not an assertion of retained tags.
    return math.fsum(p for s,p in law.items() if
        np.any(maximum_bipartite_matching(csr_matrix(m[list(s)] > 0),
                                         perm_type='column') < 0))


def tagged_replay(order, m):
    """Check whether saved cluster blocks coalesce to the retained full bars."""
    ids, shares = [], []
    for c in order.clusters:
        for b in [c.floor]:
            if b is not None and b.index is not None and b.index >= 0 and b.percentage > 0:
                ids.append(b.index); shares.append(b.percentage)
        for z in c.zones:
            ids.extend(map(int,z.indices)); shares.extend(map(float,z.shares))
        for b in [c.ceil]:
            if b is not None and b.index is not None and b.index >= 0 and b.percentage > 0:
                ids.append(b.index); shares.append(b.percentage)
    ids=np.asarray(ids);shares=np.asarray(shares)
    keep=shares > 1e-9
    ids=ids[keep]
    source_runs=ids[np.r_[True,ids[1:]!=ids[:-1]]]
    actual=order.get()[:,0].astype(int)
    actual_runs=actual[np.r_[True,actual[1:]!=actual[:-1]]]
    return (bool(np.array_equal(source_runs,actual_runs)),
            bool(np.allclose(order.pop.inclusions @ m,1,atol=1e-8,rtol=0)))


def old_builder():
    # Exact pre-patch function, not a historical execution reconstruction.
    tree = ast.parse((OUT/'source_backups/graphical_sampling/order.py').read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'Order')
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '_build_order')
    env = {'np': np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<pre-patch-order-builder>', 'exec'), env)
    return env['_build_order']


def published_comparisons(population, method, n, ip):
    result = []
    sources = [ROOT/'simulations/results/modified_summary_new_pivotal.csv',
               ROOT/'simulations/results/variance_outputs/variance_estimator_results_partial.csv']
    for source in sources:
        with source.open() as f:
            for line, r in enumerate(csv.DictReader(f), 2):
                rp = r['Population'].lower().replace('pop', '')
                ri = r.get('Probability', r.get('Mode', '')).lower()
                rm = {'initial':'NMS', 'best':'GMS'}.get(r['Method'], r['Method'])
                if (rp == population.lower() and int(float(r['n'])) == n and rm == method
                        and ri in ({'equal', 'ep'} if ip == 'EP' else {'unequal', 'up'})):
                    result.append(dict(path=str(source.relative_to(ROOT)), line=line,
                                       MI=float(r['Mm']), VI=float(r['Vm'])))
    return result


def main():
    manifest = json.loads((OUT/'historical_pickle_hashes_before.json').read_text())
    candidates = [r for r in manifest if ('best_design' in r['path'] or
                  'initial_design' in r['path'] or 'initial_desing' in r['path'])]
    candidates.sort(key=lambda r: (0 if r['path'].startswith('simulations/best_designs/')
                    else 1 if r['path'].startswith('simulations/Store/best_design/') else 2,
                    r['path']))
    seen = set()
    chosen = []
    empty = []
    for r in candidates:
        if r['size'] == 0:
            empty.append(r['path'])
        elif r['sha256'] not in seen:
            chosen.append(r)
            seen.add(r['sha256'])
    before_build = old_builder()
    records = []
    initial_memberships = {}
    loaded = {}
    for file in chosen:
        with (ROOT/file['path']).open('rb') as f:
            loaded[file['path']] = pickle.load(f)
        d = loaded[file['path']]
        if not Path(file['path']).name.startswith('best_'):
            initial_memberships[(d.pop.N,d.pop.n,np.ptp(d.pop.inclusions)<1e-12,
                                 hashlib.sha256(d.pop.coords.tobytes()).hexdigest())] = (
                                     file['path'], membership(d.order))
    for file in chosen:
        path = ROOT/file['path']
        d = loaded[file['path']]
        a = d.order.get()
        ids = a[:, 0].astype(int)
        shares = np.bincount(ids, weights=a[:, 1], minlength=d.pop.N)
        runs = ids[np.r_[True, ids[1:] != ids[:-1]]]
        law = raw_law(d)
        raw_stats = summarize(law, d.pop)
        # Historical files all have a cache. Access API only after raw audit.
        had_cache = d._all_samples_and_probs is not None
        samples, probs = d.all_samples_and_probs
        api = aggregate(zip(samples, probs))
        name = path.name.lower()
        population = ('Meuse' if 'meuse' in name else 'Swiss' if 'swiss' in name
                      else 'Aggregated' if 'aggregated' in name else 'Regular')
        ip = 'EP' if np.ptp(d.pop.inclusions) < 1e-12 else 'UP'
        method = 'GMS' if name.startswith('best_') else 'NMS'
        m = membership(d.order)
        cluster_stats = {'available': m is not None, 'clusters': len(d.order.clusters)}
        if m is not None:
            masses = d.pop.inclusions @ m
            cluster_stats.update(mass_min=float(masses.min()), mass_max=float(masses.max()),
                                 max_unit_share_error=float(np.max(np.abs(m.sum(axis=1)-1))))
            if m.shape[1] == d.pop.n:
                fraction_bad = math.fsum(p for s,p in law.items()
                    if np.max(np.abs(m[list(s)].sum(axis=0)-1)) > 1e-6)
                labels = np.argmax(m, axis=1)
                hard_bad = math.fsum(p for s,p in law.items()
                    if not np.all(np.bincount(labels[list(s)], minlength=d.pop.n) == 1))
                cluster_stats.update(fraction_weighted_count_not_one_mass=fraction_bad,
                                     hard_argmax_count_not_one_mass=hard_bad)
                runs_match, mass_one = tagged_replay(d.order,m)
                cluster_stats.update(source_cluster_blocks_coalesce_to_retained_unit_order=runs_match,
                                     source_cluster_masses_one=mass_one,
                                     no_possible_one_unit_per_soft_cluster_assignment_mass=unmatchable_mass(law,m))
        initial = initial_memberships.get((d.pop.N,d.pop.n,np.ptp(d.pop.inclusions)<1e-12,
                                          hashlib.sha256(d.pop.coords.tobytes()).hexdigest()))
        initial_comparison = None
        if method == 'GMS' and initial is not None and initial[1] is not None:
            initial_comparison = dict(path=initial[0], exact_ancestor_unproven=True,
                no_possible_one_unit_per_soft_cluster_assignment_mass=unmatchable_mass(law,initial[1]))
        rebuilt = {}
        for stage, builder in [('before', before_build), ('after', None)]:
            order = d.order.copy()
            if builder is None:
                order._build_order(order.num_splits)
            else:
                builder(order, order.num_splits)
            rebuilt_d = Design.from_order(d.pop, order)
            rebuilt_law = raw_law(rebuilt_d)
            rebuilt[stage] = summarize(rebuilt_law, d.pop)
            rebuilt[stage]['l1_from_retained_heaps'] = distance(law, rebuilt_law)
        replay = raw_law(Design.from_order(d.pop, d.order.copy()))
        # Re-score retained support, not a reconstructed hierarchy or optimized replacement.
        mi = float(np.dot(np.asarray(probs, dtype=float), Moran(d.pop).score(samples)))
        vi = float(np.dot(np.asarray(probs, dtype=float), Voronoi(d.pop).score(samples)))
        record = dict(path=file['path'], sha256=file['sha256'], population=population,
                      method=method, N=d.pop.N, n=d.pop.n, IP=ip,
                      num_splits=d.order.num_splits, order_rows=len(ids),
                      unique_unit_ids=len(np.unique(ids)),
                      count_frequencies={str(k):int(v) for k,v in Counter(np.bincount(ids)).items()},
                      contiguous_complete_unit_bars=len(runs)==d.pop.N==len(np.unique(runs)),
                      max_unit_share_error=float(np.max(np.abs(shares-1))),
                      raw=raw_stats, api=summarize(api,d.pop), cache_present=had_cache,
                      raw_api_l1=distance(law,api), retained_hierarchy=cluster_stats,
                      comparison_to_saved_initial_clusters=initial_comparison,
                      replay=summarize(replay,d.pop), replay_l1=distance(law,replay),
                      current_rebuild=rebuilt,
                      cached_MI=cache_mean(d,'_moran_expected_and_variance'),
                      cached_VI=cache_mean(d,'_voronoi_expected_and_variance'),
                      fresh_current_MI=mi, fresh_current_VI=vi,
                      retained_summary_rows=published_comparisons(population,method,d.pop.n,ip))
        records.append(record)
        print(file['path'], 'support',len(api),'raw/IP',raw_stats['probability_sum'],raw_stats['max_fip_error'],flush=True)
    after_changes = [r['path'] for r in manifest
                     if hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()!=r['sha256']]
    assert not after_changes, after_changes
    payload = dict(designs=records, empty_archives=empty,
                   pickle_files_verified_unchanged=len(manifest), changed_historical_files=after_changes)
    (OUT/'saved_design_audit.json').write_text(json.dumps(payload,indent=2)+'\n')
    fields = ['population','method','N','n','IP','num_splits','order_rows','unique_unit_ids',
              'contiguous_complete_unit_bars','support','probability_sum','max_fip_error',
              'cached_MI','cached_VI','fresh_current_MI','fresh_current_VI','path']
    with (OUT/'saved_design_audit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in records:
            row={k:r[k] for k in fields if k in r}
            row.update({k:r['api'][k] for k in ['support','probability_sum','max_fip_error']})
            writer.writerow(row)
    print('Audited',len(records),'distinct designs; unchanged pickles:',len(manifest))


if __name__ == '__main__':
    main()
