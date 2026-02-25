import json
import os
from cerebras.sdk.client import SdkLauncher

out_path = "compile_out"

# Read the compile artifact_id from the JSON file written by appliance_compile.py
with open(f"{out_path}/artifact_path.json", "r", encoding="utf8") as f:
    data = json.load(f)
    artifact_id = data["artifact_id"]

simulator = False # running on real hardware

with SdkLauncher(artifact_id, simulator=simulator, disable_version_check=True) as launcher:
    launcher.stage("run.py")

    # Run the host program. %CMADDR% is required for real hardware, not allowed for simulator.
    if simulator:
        response = launcher.run("cs_python run.py --name out")
    else:
        response = launcher.run("cs_python run.py --name out --cmaddr %CMADDR%")

    print("Host code execution response: ", response)

    if simulator:
        launcher.download_artifact("sim.log", "./output_dir/sim.log")
        print("Simulator mode on cluster completed")
    else:
        print("Appliance/Real Hardware Mode on Cluster Completed")
