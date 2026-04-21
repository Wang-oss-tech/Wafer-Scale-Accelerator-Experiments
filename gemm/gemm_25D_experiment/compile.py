import os
import sys
import json
import time
import shutil
import tempfile
from cerebras.sdk.client import SdkCompiler

P         = int(sys.argv[1])
Mt        = int(sys.argv[2])
Kt        = int(sys.argv[3])
Nt        = int(sys.argv[4])
simulator = sys.argv[5].lower() == "true"

out_path = "compile_out"

os.makedirs(out_path, exist_ok=True)

print("Start compiling: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), flush=True)

if simulator:
    ARGS = f"--arch=wse3 --fabric-dims={P+7},{P+2} --fabric-offsets=4,1 -o out --memcpy --channels=1 --params=P:{P},Mt:{Mt},Kt:{Kt},Nt:{Nt}"
else:
    ARGS = f"--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 -o out --memcpy --channels=1 --params=P:{P},Mt:{Mt},Kt:{Kt},Nt:{Nt}"

# pe.csl uses simd_32 for WSE-3 so it compiles with the SDK 1.4.0 local simulator.
# SDK 2.9.0 (hardware) rejects simd_32 for WSE-3 and requires simd_max instead.
# We patch pe.csl in a temp directory before handing it to SdkCompiler.
app_dir = tempfile.mkdtemp(prefix="summa_hw_compile_")
try:
    for fname in os.listdir("."):
        src = os.path.join(".", fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(app_dir, fname))

    pe_path = os.path.join(app_dir, "pe.csl")
    with open(pe_path, "r") as f:
        pe_src = f.read()
    # WSE-3 branches use simd_32 (local-sim-compatible); swap to simd_max for hardware.
    pe_src = pe_src.replace(
        ".simd_mode = .{ .simd_32 = true }",
        ".simd_mode = .{ .simd_max = true }",
    )
    with open(pe_path, "w") as f:
        f.write(pe_src)

    with SdkCompiler(resource_cpu=48000, resource_mem=64<<30, disable_version_check=True) as compiler:
        artifact_id = compiler.compile(
            app_path=app_dir,
            csl_main="layout.csl",
            options=ARGS,
            out_path=out_path,
        )
        with open(f"{out_path}/artifact_{P}_{Mt}_{Kt}_{Nt}.json", "w", encoding="utf-8") as f:
            json.dump({"artifact_id": artifact_id}, f)
finally:
    shutil.rmtree(app_dir, ignore_errors=True)

print("End compiling: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), flush=True)
