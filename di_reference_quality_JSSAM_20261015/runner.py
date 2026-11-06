"""Resumable pilot/full DAG with real progress and monitored CPU/affinity limits."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-jssam-reference')
from pathlib import Path
import sys,json,time,math,multiprocessing as mp,collections,argparse,traceback,shutil
import numpy as np
import psutil
from tqdm import tqdm
from threadpoolctl import threadpool_limits,threadpool_info
threadpool_limits(1)
from pipeline import HOME,PROTOCOL,population,evaluate,OT_folder,baseline_state,benchmark
from ot_solver import solve,diagnostic,atomic_json
from numerical import sinkhorn_batch,exact_open_tsp
from corrected_tsp import solve_corrected
CASES={c['scenario']:c for c in PROTOCOL['scenarios']}
RESULTS=HOME/'task_results';RESULTS.mkdir(exist_ok=True)

def task_key(t):return f'{t["scenario"]}_{t["kind"]}_{t["index"]:02d}'
def result_path(t):return RESULTS/(task_key(t)+'.json')

def prepare_cached(case,folder):
    raw,pop=population(case);cache=HOME/case['baseline_cache'];old=np.load(cache/('frozen.npz' if (cache/'frozen.npz').exists() else 'state.npz'))
    centers=old['centers'];labels=old['preliminary_labels'] if 'preliminary_labels' in old else old['labels'];P=old['P']
    assert np.array_equal(old['coords'],pop.coords) and np.array_equal(old['pi'],pop.inclusions) and np.array_equal(old['sample'],case['sample'])
    assert np.array_equal(labels,np.argmax(P,axis=1));info=diagnostic(pop.coords,pop.inclusions,centers,P)
    assert max(info['row_marginal_error'],info['column_marginal_error'])<=1e-10 and info['fixed_point_residual']<=2e-7
    np.savez_compressed(folder/'solution.npz',P=P,centers=centers,labels=labels,sample=np.array(case['sample']),initial_centers=pop.coords[case['sample']],coords=pop.coords,pi=pop.inclusions)
    info.update(converged=True,reused=True,source=str(cache.relative_to(HOME)),cpu_seconds=0.,variant=0,seed=0,beta=1.,reverse_sweep=False)
    atomic_json(folder/'diagnostics.json',info);return info

def execute(t):
    os.nice(10);start=time.perf_counter();cpu=time.process_time();case=CASES[t['scenario']];kind=t['kind'];index=t['index']
    result=dict(task=t,status='failed')
    try:
        _,pop=population(case)
        if kind in ('baseline','B'):
            folder=OT_folder(case,0 if kind=='baseline' else index);folder.mkdir(parents=True,exist_ok=True)
            if kind=='baseline' and case['baseline_cache']:
                if not (folder/'solution.npz').exists():prepare_cached(case,folder)
                info=json.loads((folder/'diagnostics.json').read_text())
            else:
                seed=0 if kind=='baseline' else PROTOCOL['alternative_seed_base']+1000*list(CASES).index(case['scenario'])+index
                ot,info=solve(pop.coords,pop.inclusions,np.array(case['sample']),folder,0 if kind=='baseline' else index,seed,
                    cpu_limit=PROTOCOL['baseline_CPU_limit'] if kind=='baseline' else PROTOCOL['alternative_CPU_limit'])
            result['OT_diagnostics']=info
            if not info['converged']:
                result.update(status='OT_failed',issue=info['error'])
            else:
                run=evaluate(case,folder,'baseline' if kind=='baseline' else 'B',index,0)
                result.update(status='completed',run=run)
                if kind=='baseline':
                    (HOME/'baselines').mkdir(exist_ok=True)
                    atomic_json(HOME/'baselines'/f'{case["scenario"]}.json',dict(ot_folder=str(folder.relative_to(HOME)),run_id=run['run_id'],selection='prespecified default'))
        else:
            folder=baseline_state(case) if kind=='A' else OT_folder(case,index)
            assert folder is not None,'Baseline unavailable'
            seed=PROTOCOL['TSP_seed_base']+1000*list(CASES).index(case['scenario'])+index+(100000 if kind=='C' else 0)
            run=evaluate(case,folder,kind,index,seed);result.update(status='completed',run=run)
    except Exception as error:
        result.update(status='failed',issue=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    result.update(cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-start)
    atomic_json(result_path(t),result)

class Controller:
    def __init__(self,jobs):
        self.jobs=jobs;self.ctx=mp.get_context('fork');self.active={};self.paused=set();self.hist=collections.deque(maxlen=15);self.target=jobs;self.last_full=False
        self.started=time.time();self.phase='pilot';self.last_status='Preparing baselines and five-run pilots'
        self.goals={(s,e):40 for s in CASES for e in 'ABC'};self.excluded=set();self.revisions=[];self.done={}
        if (HOME/'budget.json').exists():
            b=json.loads((HOME/'budget.json').read_text());self.goals={(r['scenario'],r['experiment']):r['planned'] for r in b['goals']};self.revisions=b['revisions']
        for p in RESULTS.glob('*.json'):
            r=json.loads(p.read_text());self.done[task_key(r['task'])]=r
        for r in self.done.values():
            t=r['task']
            if t['kind']=='B' and r['status']!='completed':self.excluded.add((t['scenario'],'C',t['index']))
        self.bar=tqdm(total=480,desc='Overall',bar_format='{desc} {bar} {percentage:3.0f}% | {n_fmt}/{total_fmt} resolved attempts | Elapsed {elapsed}',dynamic_ncols=True)
        self.log=(HOME/'resources.csv').open('a',buffering=1)
        if self.log.tell()==0:self.log.write('utc_epoch,cpu_percent,rolling_30s_cpu_percent,memory_percent,available_GiB,worker_RSS_GiB,workers,runnable,suspended,target,phase\n')
        psutil.cpu_percent(interval=None)
    def total(self):return sum(self.goals.values())-sum(i<=self.goals.get((s,e),0) for s,e,i in self.excluded)
    def budget(self,reason=None):
        atomic_json(HOME/'budget.json',dict(goals=[dict(scenario=s,experiment=e,planned=n) for (s,e),n in self.goals.items()],excluded=[list(x) for x in sorted(self.excluded)],revisions=self.revisions))
    def progress(self,cpu,rolling,mem,runnable):
        finished=[r for r in self.done.values() if r['task']['kind']!='baseline' and r['task']['index']<=self.goals.get((r['task']['scenario'],r['task']['kind']),0)]
        good=sum(r['status']=='completed' for r in finished);fail=len(finished)-good;total=self.total()
        groups=[]
        for (sid,e),planned in self.goals.items():
            d=[r for r in finished if r['task']['scenario']==sid and r['task']['kind']==e];excluded=sum(s==sid and ex==e and ix<=planned for s,ex,ix in self.excluded)
            groups.append(dict(scenario=sid,experiment=e,completed_attempts=len(d),successful_runs=sum(r['status']=='completed' for r in d),planned_runs=planned-excluded,failed_attempts=sum(r['status']!='completed' for r in d)))
        labels=[f'{t["scenario"]}/{t["kind"]}{t["index"]:02d}' for p,t in self.active.values()]
        eta=None
        if self.phase=='full':
            estimate=0.
            for g in groups:
                times=[r['cpu_seconds'] for r in finished if r['task']['scenario']==g['scenario'] and r['task']['kind']==g['experiment']]
                if times:estimate+=max(0,g['planned_runs']-g['completed_attempts'])*float(np.median(times))
            eta=estimate/max(1,min(self.jobs,len(self.active) or self.jobs))
        p=dict(phase=self.phase,overall_percent=100*len(finished)/max(1,total),completed_attempts=len(finished),planned_attempts=total,successful_DI_runs=good,failed_attempts=fail,
            skipped_dependent_runs=len(self.excluded),elapsed_seconds=time.time()-self.started,estimated_remaining_seconds=eta,active_workers=len(self.active),runnable_workers=len(runnable),cpu_percent=cpu,
            rolling_30s_cpu_percent=rolling,memory_percent=mem.percent,current=labels,last_status=self.last_status,groups=groups,budget_revisions=self.revisions,
            progress_definition='Completed real run attempts, including separately identified OT/other failures. Blocked/uncomputed dependent runs are excluded transparently, never counted as computations.')
        atomic_json(HOME/'progress.json',p)
        text=f'{self.phase} | Current {",".join(labels[:3]) or "idle"} | valid DI {good}, failed {fail} | workers {len(runnable)}/{self.jobs} | CPU {cpu:.1f}% | ETA '+('pilot pending' if eta is None else f'~{eta/60:.1f}m')
        self.bar.total=total;self.bar.n=len(finished);self.bar.set_postfix_str(text,refresh=True)
        (HOME/'progress.txt').write_text(f'Overall {p["overall_percent"]:.1f}% | {len(finished)}/{total} actual attempts | validDI {good} | failed {fail}\n'+text+'\n'+ '\n'.join(f'{g["scenario"]} {g["experiment"]}: {g["completed_attempts"]}/{g["planned_runs"]} attempts, {g["successful_runs"]} valid' for g in groups)+'\n')
    def run_stage(self,phase):
        self.phase=phase;queue=collections.deque();queued=set();self.phase_limit={s:5 if phase=='pilot' else self.goals[(s,'B')] for s in CASES}
        def add(s,e,i):
            t=dict(scenario=s,kind=e,index=i);key=task_key(t)
            if key not in self.done and key not in queued and (s,e,i) not in self.excluded:queue.append(t);queued.add(key)
        if phase=='pilot':
            for s in CASES:
                if baseline_state(CASES[s]) is None:add(s,'baseline',0)
        for i in range(1,41):
            for s in CASES:
                limit=5 if phase=='pilot' else self.goals[(s,'B')]
                if i<=limit:add(s,'B',i)
        def dependencies():
            for s,c in CASES.items():
                base=baseline_state(c)
                if base is None:
                    br=self.done.get(f'{s}_baseline_00')
                    complete=all(f'{s}_B_{i:02d}' in self.done for i in range(1,6))
                    if br is not None and complete:
                        valid=[i for i in range(1,6) if self.done[f'{s}_B_{i:02d}']['status']=='completed']
                        if valid:
                            i=min(valid);folder=OT_folder(c,i);self.last_status=f'{s}: default baseline failed; lowest prespecified converged pilot {i} becomes frozen baseline'
                            base_run=evaluate(c,folder,'baseline_fallback',0,0)
                            (HOME/'baselines').mkdir(exist_ok=True);atomic_json(HOME/'baselines'/f'{s}.json',dict(ot_folder=str(folder.relative_to(HOME)),run_id=base_run['run_id'],selection=self.last_status))
                            self.revisions.append(dict(reason=self.last_status,time=time.time()));base=folder
                        else:
                            for i in range(1,self.goals[(s,'A')]+1):self.excluded.add((s,'A',i))
                if base is not None:
                    for i in range(1,(5 if phase=='pilot' else self.goals[(s,'A')])+1):add(s,'A',i)
                for i in range(1,(5 if phase=='pilot' else self.goals[(s,'B')])+1):
                    r=self.done.get(f'{s}_B_{i:02d}')
                    if r is not None:
                        if r['status']=='completed' and i<=self.goals[(s,'C')]:add(s,'C',i)
                        elif r['status']!='completed':self.excluded.add((s,'C',i))
        dependencies();self.budget()
        while queue or self.active:
            for pid,(proc,t) in list(self.active.items()):
                if not proc.is_alive():
                    proc.join();del self.active[pid];self.paused.discard(pid)
                    path=result_path(t)
                    if path.exists():r=json.loads(path.read_text())
                    else:r=dict(task=t,status='worker_exit',issue=f'exit {proc.exitcode}',cpu_seconds=0.);atomic_json(path,r)
                    self.done[task_key(t)]=r;self.last_status=f'{task_key(t)}: {r["status"]}';dependencies();self.budget()
            cpu=psutil.cpu_percent(interval=None);self.hist.append(cpu);rolling=float(np.mean(self.hist));mem=psutil.virtual_memory()
            if cpu>=58 or rolling>=57.5 or mem.percent>=78:self.target=max(1,self.target-1)
            elif not self.last_full and cpu<50 and rolling<50 and self.target<self.jobs:self.target+=1
            full=cpu>=60 or rolling>=57.5 or mem.percent>=78
            if not full:
                while queue and len(self.active)<self.target:
                    t=queue.popleft()
                    if task_key(t) in self.done or (t['scenario'],t['kind'],t['index']) in self.excluded:continue
                    proc=self.ctx.Process(target=execute,args=(t,));proc.start();self.active[proc.pid]=(proc,t)
            runnable=set() if full else set(list(self.active)[:self.target]);rss=0.
            for pid in self.active:
                try:
                    p=psutil.Process(pid);rss+=p.memory_info().rss
                    if pid in runnable and pid in self.paused:p.resume();self.paused.remove(pid)
                    elif pid not in runnable and pid not in self.paused:p.suspend();self.paused.add(pid)
                except psutil.NoSuchProcess:pass
            self.log.write(f'{time.time()},{cpu},{rolling},{mem.percent},{mem.available/2**30},{rss/2**30},{len(self.active)},{len(runnable)},{len(self.paused)},{self.target},{phase}\n')
            self.progress(cpu,rolling,mem,runnable);self.last_full=full;time.sleep(2)
        self.progress(psutil.cpu_percent(),float(np.mean(self.hist)),psutil.virtual_memory(),set());self.budget()
    def decide(self):
        for s in CASES:
            rs=[self.done[f'{s}_B_{i:02d}'] for i in range(1,6)];valid=sum(r['status']=='completed' for r in rs);median=float(np.median([r['cpu_seconds'] for r in rs]))
            target=40
            if valid==0:target=5
            elif median>240 or valid<2:target=5
            elif median>120 or (s=='S100' and median*40>1800) or (s=='S100' and valid<4):target=10
            old=self.goals[(s,'B')];self.goals[(s,'B')]=target;self.goals[(s,'C')]=target
            if old!=target:self.revisions.append(dict(scenario=s,old_BC_target=old,new_BC_target=target,valid_pilots=valid,median_pilot_cpu=median,projected_40_OT_CPU_seconds=40*median,reason='Prespecified feasibility/cost rule; no DI outcome used',time=time.time()))
        self.budget()
    def close(self):
        self.phase='complete';self.progress(psutil.cpu_percent(),float(np.mean(self.hist)),psutil.virtual_memory(),set());self.log.close();self.bar.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pilot','full','all'],default='all');parser.add_argument('--jobs',type=int,default=8);args=parser.parse_args()
    available=sorted(os.sched_getaffinity(0));affinity=available[:max(1,math.floor(len(available)*.60))];os.sched_setaffinity(0,affinity)
    jobs=min(args.jobs,max(1,int(len(available)*.55)))
    atomic_json(HOME/'resource_policy.json',dict(available_CPUs=available,affinity_CPUs=affinity,max_workers=jobs,threads_per_worker=1,thread_pools=threadpool_info(),target_total_CPU='50–55%',hard_pause_total_CPU=60,rolling_pause_CPU=57.5))
    # Compile once before fork; warm-up changes no scientific seed or sample.
    sinkhorn_batch(np.zeros((4,2)),np.full(4,.5),np.zeros(4),np.zeros(2),.01,1,False)
    solve_corrected(np.array([[0.,0.],[.3,.4],[.7,.9]]),np.arange(3),np.arange(1,4),np.arange(1,4),0,2)
    exact_open_tsp(np.array([[0.,0.],[.3,.4],[.7,.9]]))
    c=Controller(jobs)
    try:
        if args.stage in ('pilot','all'):c.run_stage('pilot');c.decide()
        if args.stage in ('full','all'):c.run_stage('full')
        if args.stage=='all' or args.stage=='full':c.close()
        else:c.bar.close();c.log.close()
    except BaseException:
        for pid,(proc,t) in c.active.items():
            try:psutil.Process(pid).resume();proc.terminate()
            except psutil.NoSuchProcess:pass
        c.bar.close();c.log.close();raise

if __name__=='__main__':main()
