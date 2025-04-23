# GenAI is used for rephrasing comments and debugging.
import os
import subprocess
from itertools import product
import time
from datetime import datetime

# Define all options
inits = ["random", "mocov2", "imagenet"]
cams = ["gradcam", "cam"]
weak_supervisions = ["weak_gradcam", "weak_cam"]
full_supervision = ["full"]

# Path to main script
MAIN_SCRIPT = "main.py"

# Create directories for logs
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create a timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def run_experiment(command, experiment_name):
    """Helper function to run a single experiment"""
    log_path = f"{LOG_DIR}/{experiment_name}_{timestamp}.log"
    print(f"\nRunning experiment: {experiment_name}")
    print(f"Command: {' '.join(command)}")
    print(f"Log file: {log_path}")
    
    start_time = time.time()
    try:
        # Run the command and show output in real-time while also saving to log
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Print output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    log_file.write(output)
                    log_file.flush()
            
            return_code = process.poll()
        
        duration = time.time() - start_time
        if return_code == 0:
            print(f"✅ Experiment completed in {duration:.2f} seconds")
            return True
        else:
            print(f"❌ Experiment failed with return code {return_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error running experiment: {str(e)}")
        return False

# --- Run weakly-supervised combinations (with CAM)
cam_supervision_pairs = {
    "gradcam": "weak_gradcam",
    "cam": "weak_cam"
}

for init in inits:
    for cam, supervision in cam_supervision_pairs.items():
        experiment_name = f"{init}_{cam}_{supervision}"
        command = [
            "python", MAIN_SCRIPT, "run_all",
            "--init", init,
            "--cam", cam,
            "--supervision", supervision
        ]
        if not run_experiment(command, experiment_name):
            print(f"❌ Failed to run {experiment_name}")

# --- Run fully-supervised combinations (NO CAM)
for init in inits:
    supervision = "full"
    experiment_name = f"{init}_{supervision}"
    command = [
        "python", MAIN_SCRIPT, "run_all",
        "--init", init,
        "--supervision", supervision
    ]
    if not run_experiment(command, experiment_name):
        print(f"❌ Failed to run {experiment_name}")

print(f"\n✅ All experiments finished. Check {LOG_DIR}/ for logs")
