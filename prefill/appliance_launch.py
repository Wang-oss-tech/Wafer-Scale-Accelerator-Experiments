import argparse
import json
from cerebras.sdk.client import SdkLauncher

out_path = "compile_out"


def parse_args():
    parser = argparse.ArgumentParser(description="WSE-3 Prefill appliance launcher")
    parser.add_argument("--P",          required=True, type=int,  help="PE grid size P x P")
    parser.add_argument("--dim",        required=True, type=int,  help="Model hidden dimension")
    parser.add_argument("--seq_len",    required=True, type=int,  help="Sequence length")
    parser.add_argument("--ffn_dim",    required=True, type=int,  help="FFN intermediate dimension")
    parser.add_argument("--head_dim",   required=True, type=int,  help="Attention head dimension")
    parser.add_argument("--n_heads",    default=1,     type=int,  help="Number of attention heads (default 1)")
    parser.add_argument("--n_kv_heads", default=1,     type=int,  help="Number of KV heads (default 1)")
    parser.add_argument("--warmup",     default=5,     type=int,  help="Warmup runs (default 5)")
    parser.add_argument("--repeats",    default=50,    type=int,  help="Timed runs (default 50)")
    parser.add_argument("--simulator",  action="store_true",      help="Run on simulator")
    return parser.parse_args()


args = parse_args()
P          = args.P
dim        = args.dim
seq_len    = args.seq_len
ffn_dim    = args.ffn_dim

artifact_json = f"{out_path}/artifact_{P}_{dim}_{seq_len}_{ffn_dim}.json"
with open(artifact_json, "r", encoding="utf-8") as f:
    artifact_id = json.load(f)["artifact_id"]

with SdkLauncher(artifact_id, simulator=args.simulator, disable_version_check=True) as launcher:
    launcher.stage("run.py")

    run_args = (
        f"--name out"
        f" --P {P}"
        f" --dim {dim}"
        f" --seq_len {seq_len}"
        f" --ffn_dim {ffn_dim}"
        f" --head_dim {args.head_dim}"
        f" --n_heads {args.n_heads}"
        f" --n_kv_heads {args.n_kv_heads}"
        f" --warmup {args.warmup}"
        f" --repeats {args.repeats}"
    )

    if args.simulator:
        response = launcher.run(f"cs_python run.py {run_args}")
    else:
        response = launcher.run(f"cs_python run.py {run_args} --cmaddr %CMADDR%")

    print("Host code execution response: ", response)

    if args.simulator:
        launcher.download_artifact("sim.log", "./output_dir/sim.log")
        print("Simulator mode on cluster completed")
    else:
        print("Appliance/Real Hardware Mode on Cluster Completed")
