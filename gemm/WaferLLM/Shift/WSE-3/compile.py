import sys
import json
import time
from cerebras.sdk.client import SdkCompiler

P = int(sys.argv[1])
L = int(sys.argv[2])
Mt = int(sys.argv[3])
Nt = int(sys.argv[4])

simulator = sys.argv[5].lower()=="true"

out_path = "compile_out"

print("Start compiling: "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)

if simulator:
    ARGS=f"--arch=wse3 --fabric-dims={P+L+7},{P+2} --fabric-offsets=4,1 -o out --memcpy --channels=1 --params=P:{P},L:{L},Mt:{Mt},Nt:{Nt}"
else:
    ARGS=f"--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 -o out --memcpy --channels=1 --params=P:{P},L:{L},Mt:{Mt},Nt:{Nt}"

# Instantiate compiler
with SdkCompiler(resource_cpu=48000, resource_mem=64<<30, disable_version_check=True) as compiler:

    # Launch compile job
    artifact_id = compiler.compile(
        app_path="src",
        csl_main="layout.csl",
        options=ARGS,
        out_path=out_path,
    )

    # Write the artifact_id to a JSON file
    with open(f"{out_path}/artifact_{P}_{L}_{Mt}_{Nt}.json", "w", encoding="utf-8") as f:
        json.dump({"artifact_id": artifact_id,}, f)

print("End compiling: "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)