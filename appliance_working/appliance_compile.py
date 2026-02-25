import json
import os
import time
from cerebras.sdk.client import SdkCompiler

out_path = "compile_out"
os.makedirs(out_path, exist_ok=True)

# Instantiate copmiler using a context manager
# Disable version check to ignore appliance client and server version differences.
with SdkCompiler(disable_version_check=True) as compiler:

    # Launch compile job
    artifact_id = compiler.compile(
        app_path=".",
        csl_main="layout.csl",
        options="--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 --memcpy --channels=1 -o out",  # real hardware
        # "--arch=wse3  --fabric-dims=8,3 --fabric-offsets=4,1 --memcpy --channels=1 -o out",    # simulator mode
        out_path=out_path
    )

    # Write the artifact_path to a JSON file (use absolute path for robustness)
    with open(f"{out_path}/artifact_path.json", "w", encoding="utf8") as f:
        json.dump({"artifact_id": artifact_id,}, f)

    print("End compiling: "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)


 