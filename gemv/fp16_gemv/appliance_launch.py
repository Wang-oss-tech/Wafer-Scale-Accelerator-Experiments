import argparse
import json
from cerebras.sdk.client import SdkLauncher

out_path = "compile_out"


def parse_args():
    parser = argparse.ArgumentParser(description="fp16 GEMV appliance launcher")
    parser.add_argument("--P",         required=True, type=int, help="PE grid size P x P")
    parser.add_argument("--M",         required=True, type=int, help="Matrix rows (output dim)")
    parser.add_argument("--N",         required=True, type=int, help="Matrix cols (input dim)")
    parser.add_argument("--simulator", action="store_true",     help="Run on simulator")
    return parser.parse_args()


args = parse_args()
P = args.P
M = args.M
N = args.N

artifact_json = f"{out_path}/artifact_{P}_{M}_{N}.json"
with open(artifact_json, "r", encoding="utf-8") as f:
    artifact_id = json.load(f)["artifact_id"]

with SdkLauncher(artifact_id, simulator=args.simulator, disable_version_check=True) as launcher:
    launcher.stage("run_hw.py")
    launcher.stage("compile_out")

    run_args = f"--P {P} --M {M} --N {N}"
    if args.simulator:
        response = launcher.run(f"cs_python run_hw.py {run_args} --simulator")
        launcher.download_artifact("sim.log", "./sim.log")
    else:
        response = launcher.run(f"cs_python run_hw.py {run_args}")

    print("Host code execution response: ", response)
