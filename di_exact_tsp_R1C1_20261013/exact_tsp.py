"""Exhaustive optimal open TSP with reversal-aware deterministic tie handling."""
from decimal import Decimal,localcontext
import itertools
import numpy as np
from scipy.spatial.distance import cdist

PRECISION=80
TIE_TOLERANCE=Decimal('1e-60')

def decimal_distances(points,precision=PRECISION):
    with localcontext() as ctx:
        ctx.prec=precision
        coords=[[Decimal.from_float(float(c)) for c in row] for row in points]
        return [[sum((a-b)**2 for a,b in zip(coords[i],coords[j])).sqrt() for j in range(len(points))] for i in range(len(points))]

def path_cost(path,distances,precision=PRECISION):
    # Sum edges in a canonical order so reversal has identical rounding.
    edges=sorted(tuple(sorted((int(a),int(b)))) for a,b in zip(path[:-1],path[1:]))
    with localcontext() as ctx:
        ctx.prec=precision
        return sum((distances[a][b] for a,b in edges),Decimal(0))

def enumerate_paths(points):
    points=np.asarray(points,float);n=len(points)
    if n<2:raise ValueError('This focused solver requires at least2 distinct centroids.')
    if len(np.unique(points,axis=0))!=n:raise ValueError('Coincident centroids require an additional stable node key; not silently resolved.')
    order=np.lexsort(tuple(points[:,j] for j in range(points.shape[1]-1,-1,-1)))
    # np.lexsort's last key is primary: reversed key tuple makes x primary.
    rank_points=points[order];distances=decimal_distances(rank_points)
    records=[]
    for path in itertools.permutations(range(n)):
        if path>path[::-1]:continue
        records.append((path,path_cost(path,distances)))
    best=min(value for _,value in records)
    ties=sorted(path for path,value in records if abs(value-best)<=TIE_TOLERANCE)
    higher=decimal_distances(rank_points,120)
    high_costs=[(path,path_cost(path,higher,120)) for path in ties]
    highbest=min(value for _,value in high_costs)
    high_ties=sorted(path for path,value in high_costs if abs(value-highbest)<=TIE_TOLERANCE)
    chosen=high_ties[0]
    other=[value for path,value in records if path not in high_ties]
    gap=min(other)-best if other else None
    # Independent ordinary-precision full oriented enumeration checks the winner.
    D=cdist(rank_points,rank_points);allpaths=np.array(list(itertools.permutations(range(n))),int)
    lengths=D[allpaths[:,:-1],allpaths[:,1:]].sum(axis=1)
    assert abs(float(best)-float(lengths.min()))<1e-12
    info=dict(centroid_labels_in_spatial_rank_order=order.tolist(),canonical_paths_evaluated=len(records),oriented_paths_covered=len(allpaths),best_length_decimal=str(best),high_precision_best_length_decimal=str(highbest),precision_digits=PRECISION,check_precision_digits=120,numerical_tie_tolerance=str(TIE_TOLERANCE),optimal_unoriented_paths=[list(p) for p in high_ties],optimal_oriented_paths_count=2*len(high_ties),chosen_spatial_rank_path=list(chosen),chosen_original_label_path=order[np.array(chosen)].tolist(),runner_up_unoriented_length_gap_decimal=str(gap) if gap is not None else None)
    return info,records

class ExactOpenTSPSolver:
    def __init__(self,oriented_optimal_index=0):
        self.oriented_optimal_index=oriented_optimal_index
        self.info=None
    def solve(self,points,restarts=15):
        self.info,self.records=enumerate_paths(points)
        order=np.array(self.info['centroid_labels_in_spatial_rank_order'],int)
        oriented=[]
        for p in self.info['optimal_unoriented_paths']:
            oriented.extend([p,list(reversed(p))])
        if not 0<=self.oriented_optimal_index<len(oriented):raise ValueError('Invalid optimal-path index')
        self.selected_spatial_rank_path=oriented[self.oriented_optimal_index]
        return order[np.array(self.selected_spatial_rank_path,int)]
