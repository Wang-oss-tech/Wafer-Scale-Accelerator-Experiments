import os
import sys
import json
import time
import shutil
import tempfile
from cerebras.sdk.client import SdkCompiler

P            = int(sys.argv[1])
dim          = int(sys.argv[2])
seq_len      = int(sys.argv[3])
ffn_dim      = int(sys.argv[4])
n_heads      = int(sys.argv[5])
n_kv_heads   = int(sys.argv[6])
head_dim     = int(sys.argv[7])
simulator    = sys.argv[8].lower() == "true"

dim_p_pe      = dim // P
seq_len_p_pe  = seq_len // P
ffn_dim_p_pe  = ffn_dim // P
pes_p_head    = P // n_heads
pes_p_kv_head = P // n_kv_heads
head_dim_p_pe = head_dim // P

out_path = "compile_out"
os.makedirs(out_path, exist_ok=True)

print("Start compiling: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), flush=True)

params = (
    f"P:{P},dim_p_pe:{dim_p_pe},"
    f"pes_p_head:{pes_p_head},pes_p_kv_head:{pes_p_kv_head},"
    f"head_dim_p_pe:{head_dim_p_pe},"
    f"seq_len_p_pe:{seq_len_p_pe},ffn_dim_p_pe:{ffn_dim_p_pe}"
)

if simulator:
    fabric = f"--fabric-dims={P+7},{P+2} --fabric-offsets=4,1"
else:
    fabric = "--fabric-dims=762,1172 --fabric-offsets=4,1"

ARGS = f"--arch=wse3 {fabric} -o out --memcpy --channels=1 --params={params}"

# For hardware: patch simd_32/simd_64 → simd_max in comm_pe.csl
app_dir = tempfile.mkdtemp(prefix="prefill_hw_compile_")
try:
    for fname in os.listdir("."):
        src = os.path.join(".", fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(app_dir, fname))
    src_dir = os.path.join(app_dir, "src")
    os.makedirs(src_dir, exist_ok=True)
    comm_dir = os.path.join(src_dir, "comm_lib")
    os.makedirs(comm_dir, exist_ok=True)
    for sub in ["src", os.path.join("src", "comm_lib")]:
        src_sub = os.path.join(".", sub)
        dst_sub = os.path.join(app_dir, sub)
        if os.path.isdir(src_sub):
            for fname in os.listdir(src_sub):
                fp = os.path.join(src_sub, fname)
                if os.path.isfile(fp):
                    shutil.copy2(fp, os.path.join(dst_sub, fname))

    if not simulator:
        for csl_rel in [os.path.join("src", "comm_lib", "comm_pe.csl")]:
            csl_path = os.path.join(app_dir, csl_rel)
            if os.path.exists(csl_path):
                with open(csl_path, "r") as f:
                    src_text = f.read()
                src_text = src_text.replace(".simd_mode = .{ .simd_32 = true }", ".simd_mode = .{ .simd_max = true }")
                src_text = src_text.replace(".simd_mode = .{ .simd_64 = true }", ".simd_mode = .{ .simd_max = true }")
                with open(csl_path, "w") as f:
                    f.write(src_text)

    with SdkCompiler(resource_cpu=48000, resource_mem=64 << 30, disable_version_check=True) as compiler:
        artifact_id = compiler.compile(
            app_path=app_dir,
            csl_main=os.path.join("src", "layout.csl"),
            options=ARGS,
            out_path=out_path,
        )
        tag = f"{P}_{dim}_{seq_len}_{ffn_dim}"
        with open(f"{out_path}/artifact_{tag}.json", "w", encoding="utf-8") as f:
            json.dump({"artifact_id": artifact_id}, f)
finally:
    shutil.rmtree(app_dir, ignore_errors=True)

print("End compiling: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), flush=True)
