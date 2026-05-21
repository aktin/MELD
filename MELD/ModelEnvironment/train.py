import os
import subprocess

import pandas as pd

def run_training(input_data: pd.DataFrame, artifact_path: str):
    train_path = "/home/shuening/code/KlimaNot/examples/tfdf/train"
    model_py_path = os.path.join(train_path, ".venv", "bin", "python")
    if not os.path.exists(artifact_path):
        os.makedirs(artifact_path)
    input_data.to_csv(index=False, path_or_buf=os.path.join(train_path, "input.csv"))
    subprocess.run([model_py_path, "train.py"],
                               cwd=train_path)
