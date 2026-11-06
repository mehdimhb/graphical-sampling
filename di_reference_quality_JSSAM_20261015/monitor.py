"""Independent terminal view: python monitor.py (Ctrl-C exits only this monitor)."""
from pathlib import Path
import json,time,os
p=Path(__file__).resolve().parent/'progress.json'
try:
 while True:
  if p.exists():
   d=json.loads(p.read_text());print('\033[2J\033[H',end='');print(f'Overall [{"█"*int(d["overall_percent"]/5):░<20}] {d["overall_percent"]:.1f}% | {d["completed_attempts"]}/{d["planned_attempts"]} real attempts')
   print(f'Stage {d["phase"]} | valid DI {d["successful_DI_runs"]}, failed attempts {d["failed_attempts"]} | CPU {d["cpu_percent"]:.1f}% | active workers {d["runnable_workers"]}/{d["active_workers"]} | elapsed {d["elapsed_seconds"]/60:.1f}m')
   print('Current:',', '.join(d['current']) or d['last_status']);print('ETA:', 'pilot pending' if d['estimated_remaining_seconds'] is None else f'~{d["estimated_remaining_seconds"]/60:.1f}m')
   for g in d['groups']:print(f'{g["scenario"]:5s} {g["experiment"]} | {g["completed_attempts"]:2d}/{g["planned_runs"]:2d} attempts | {g["successful_runs"]:2d} valid, {g["failed_attempts"]:2d} failed')
   if d['phase']=='complete':break
  else:print('Waiting for runner progress.json…')
  time.sleep(2)
except KeyboardInterrupt:pass
