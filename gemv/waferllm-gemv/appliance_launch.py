import argparse
import json
from cerebras.sdk.client import SdkLauncher

out_path = "compile_out"


def parse_args():
    parser = argparse.ArgumentParser(description="MeshGEMV appliance launcher")
    parser.add_argument("--P",         required=True, type=int, help="PE grid size P x P")
    parser.add_argument("--M",         required=True, type=int, help="Vector dimension M (1 x M)")
    parser.add_argument("--N",         required=True, type=int, help="Matrix dimension N (M x N)")
    parser.add_argument("--group_num", default=4,     type=int, help="Allreduce group count (default 4)")
    parser.add_argument("--warmup",    default=10,    type=int, help="Warmup runs (default 10)")
    parser.add_argument("--repeats",   default=100,   type=int, help="Timed runs (default 100)")
    parser.add_argument("--simulator", action="store_true",     help="Run on simulator")
    return parser.parse_args()


args = parse_args()
P         = args.P
M         = args.M
N         = args.N
Mt        = M // P
Nt        = N // P
group_num = args.group_num

artifact_json = f"{out_path}/artifact_{P}_{Mt}_{Nt}_{group_num}.json"
with open(artifact_json, "r", encoding="utf-8") as f:
    artifact_id = json.load(f)["artifact_id"]

with SdkLauncher(artifact_id, simulator=args.simulator, disable_version_check=True) as launcher:
    launcher.stage("launch_wse3.py")

    run_args = f"--P {P} --M {M} --N {N} --group_num {group_num} --warmup {args.warmup} --repeats {args.repeats}"

    if args.simulator:
        response = launcher.run(f"cs_python launch_wse3.py {run_args} --simulator")
        launcher.download_artifact("sim.log", "./sim.log")
        print("Simulator mode on cluster completed")
    else:
        response = launcher.run(f"cs_python launch_wse3.py {run_args}")
        print("Hardware mode on cluster completed")

    print("Host code execution response: ", response)
