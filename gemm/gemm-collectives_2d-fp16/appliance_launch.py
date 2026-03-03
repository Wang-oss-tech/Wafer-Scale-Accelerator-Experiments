import argparse
import json
from cerebras.sdk.client import SdkLauncher

out_path = "compile_out"


def parse_args():
    parser = argparse.ArgumentParser(description="SUMMA fp16 appliance launcher")
    parser.add_argument("--P",       required=True, type=int, help="PE grid size P x P")
    parser.add_argument("--M",       required=True, type=int, help="Full matrix M dimension")
    parser.add_argument("--K",       required=True, type=int, help="Full matrix K dimension")
    parser.add_argument("--N",       required=True, type=int, help="Full matrix N dimension")
    parser.add_argument("--warmup",  default=5,  type=int,   help="Warmup runs (default 5)")
    parser.add_argument("--repeats", default=50, type=int,   help="Timed runs (default 50)")
    parser.add_argument("--simulator", action="store_true",  help="Run on simulator")
    return parser.parse_args()


args = parse_args()
P  = args.P
M  = args.M
K  = args.K
N  = args.N
Mt = M // P
Kt = K // P
Nt = N // P

artifact_json = f"{out_path}/artifact_{P}_{Mt}_{Kt}_{Nt}.json"
with open(artifact_json, "r", encoding="utf-8") as f:
    artifact_id = json.load(f)["artifact_id"]

with SdkLauncher(artifact_id, simulator=args.simulator, disable_version_check=True) as launcher:
    launcher.stage("run.py")

    run_args = f"--name out --P {P} --M {M} --K {K} --N {N} --warmup {args.warmup} --repeats {args.repeats}"

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
