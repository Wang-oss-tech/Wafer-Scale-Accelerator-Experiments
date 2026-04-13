import sys
import json
import time
from cerebras.sdk.client import SdkCompiler

P = int(sys.argv[1])
M = int(sys.argv[2])
N = int(sys.argv[3])

simulator = sys.argv[4].lower() == "true"

out_path = "compile_out"

print("Start compiling: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)

if simulator:
    ARGS = f"--arch=wse3 --fabric-dims={P+7},{P+2} --fabric-offsets=4,1 -o out --memcpy --channels=1 --params=kernel_rows:{P},kernel_cols:{P},matrix_rows:{M},matrix_cols:{N}"
else:
    ARGS = f"--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 -o out --memcpy --channels=1 --params=kernel_rows:{P},kernel_cols:{P},matrix_rows:{M},matrix_cols:{N}"

with SdkCompiler(resource_cpu=48000, resource_mem=64<<30) as compiler:
    artifact_id = compiler.compile(
        app_path=".",
        csl_main="layout.csl",
        options=ARGS,
        out_path=out_path,
    )

    with open(f"{out_path}/artifact_{P}_{M}_{N}.json", "w", encoding="utf-8") as f:
        json.dump({"artifact_id": artifact_id}, f)

print("End compiling: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)
