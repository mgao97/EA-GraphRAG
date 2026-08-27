import os, shutil, subprocess

ROOT = "/home/user/GSK/mgao/EA-GraphRAG"
base = os.path.join(ROOT, "baseline/graphrag/2wikimultihopqa")
# 彻底清空，使其像 hotpotqa 一样全新索引（无 graphrag_storage 残留 -> 不触发 safe-delete）
for d in ("output", "graphrag_storage", ".storage", "input"):
    p = os.path.join(base, d)
    if os.path.isdir(p):
        shutil.rmtree(p)
        print(f"[fix] removed {p}", flush=True)

cmd = [
    "/home/user/.conda/envs/graphrag/bin/python", "run_dataset.py",
    "--dataset", "2wikimultihopqa", "--n", "50",
    "--output", os.path.join(ROOT, "results/baseline_graphrag_2wikimultihopqa.csv"),
]
env = dict(os.environ)
env["GRAPHRAG_SAFE_DELETE_BULK_CONFIRM_REQUIRED"] = "False"
print("[fix] launching 2wiki index+query (fresh)", flush=True)
subprocess.run(cmd, cwd=os.path.join(ROOT, "baseline/graphrag"), env=env, check=False)
print("[fix] 2wiki done", flush=True)
