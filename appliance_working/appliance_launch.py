import json
import os
from cerebras.sdk.client import SdkLauncher

out_path = "compile_out"

# read the compile artifact_path from the json file
with open(f"{out_path}/artifact_path.json", "r", encoding="utf8") as f:
    data = json.load(f)
    artifact_id = data["artifact_id"]


simulator = False  

with SdkLauncher(artifact_id, simulator=simulator, disable_version_check=True) as launcher:
    
    launcher.stage("additional_artifact.txt")
    response = launcher.run(
        "echo \"ABOUT TO RUN IN THE APPLIANCE\"",
        "cat additional_artifact.txt",
    )

    print("Test response: ", response)
    launcher.stage("run.py")

    # Run the original host code. %CMADDR% is required for real hardware, not allowed for simulator.
    if simulator:
        response = launcher.run("cs_python run.py --name out")
    else:
        response = launcher.run("cs_python run.py --name out --cmaddr %CMADDR%")
    
    print("Host code execution response: ", response)

    if simulator:
        launcher.download_artifact("sim.log", "./output_dir/sim.log")
        print("Simulator mode on cluster completed")
    else:
        print("Appliance/Read Hardware Mode on Cluster Completed")
        # response = launcher.run("ls -la", "ls -la out")
        # print ("List response: ", response)
        # print ("--------------------------------")
        # response = launcher.run("cat additional_artifact.txt")
        # print ("Additional artifact content: ", response)
        # print ("--------------------------------")
        # response = launcher.run("cat out/out.json")
        # print ("Out.json content: ", response)
        # print ("--------------------------------")
        # response = launcher.run("cat fabric.json")
        # print ("Fabric.json content: ", response)
        # print ("--------------------------------")
        # response = launcher.run("ls -la cs_hio_config_files")
        # print ("cs_hio_config_files content: ", response)
        






    