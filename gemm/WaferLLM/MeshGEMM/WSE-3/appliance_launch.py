import argparse
import json
from cerebras.sdk.client import SdkLauncher

out_path = "compile_out"


def parse_args():
    parser = argparse.ArgumentParser(description="MeshGEMM appliance launcher")
    parser.add_argument("--P", required=True, type=int, help="PE grid size P x P")
    parser.add_argument("--M", required=True, type=int, help="Input context length")
    parser.add_argument("--K", required=True, type=int, help="Word vector dimension")
    parser.add_argument("--N", required=True, type=int, help="Output dimension")
    parser.add_argument("--simulator", action="store_true", help="Run on simulator")
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
    data = json.load(f)
    artifact_id = data["artifact_id"]

simulator = args.simulator

with SdkLauncher(artifact_id, simulator=simulator, disable_version_check=True) as launcher:

    # Stage run.py which uses sdkruntimepybind (available on the worker node).
    # launch_wse3.py used cerebras.sdk.client which is not installed on the worker.
    launcher.stage("run.py")

    run_args = f"--name out --P {P} --M {M} --K {K} --N {N}"

    # run.py uses sdkruntimepybind.SdkRuntime(name, cmaddr=...) so --cmaddr is
    # required for real hardware; not allowed for simulator.
    if simulator:
        response = launcher.run(f"cs_python run.py {run_args}")
    else:
        response = launcher.run(f"cs_python run.py {run_args} --cmaddr %CMADDR%")

    print("Host code execution response: ", response)

    if simulator:
        launcher.download_artifact("sim.log", "./output_dir/sim.log")
        print("Simulator mode on cluster completed")
    else:
        print("Appliance/Real Hardware Mode on Cluster Completed")
