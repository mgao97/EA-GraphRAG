#!/usr/bin/env bash
cd /home/user/GSK/mgao/EA-GraphRAG
echo "=== 组A结果(含EM/F1) ==="
python3 -c "
import csv,glob
for f in sorted(glob.glob('results/*.csv')):
    if 'minimax' in f: continue
    try:
        r=list(csv.DictReader(open(f)))
        if not r: continue
        em=[float(x['em']) for x in r if x.get('em') not in (None,'')]
        f1=[float(x['f1']) for x in r if x.get('f1') not in (None,'')]
        print(f'{len(r):3d} | EM={sum(em)/len(em):.3f} F1={sum(f1)/len(f1):.3f} | {f.split(\"/\")[-1]}')
    except Exception as e: print('ERR',f,e)
"
echo "=== 组B结果 ==="
python3 -c "
import csv,glob
for f in sorted(glob.glob('results/*minimax*.csv')):
    try:
        r=list(csv.DictReader(open(f)))
        if not r: continue
        em=[float(x['em']) for x in r if x.get('em') not in (None,'')]
        f1=[float(x['f1']) for x in r if x.get('f1') not in (None,'')]
        print(f'{len(r):3d} | EM={sum(em)/len(em):.3f} F1={sum(f1)/len(f1):.3f} | {f.split(\"/\")[-1]}')
    except Exception as e: print('ERR',f,e)
"
echo "=== 运行中进程 ==="
ps aux | grep -E "run_ea_only|run_dataset|graphrag index|main.py --dataset" | grep -v grep | wc -l
