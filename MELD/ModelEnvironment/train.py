import os
import subprocess

import pandas as pd


def run_training(input_data: pd.DataFrame, artifact_path: str = "artifact"):
    model_py_path = os.path.join("..", artifact_path, ".venv", "bin", "python")
    input_data.to_csv(index=False, path_or_buf=os.path.join(artifact_path, "input.csv"))
    subprocess.run([model_py_path, "train.py"],
                               cwd=artifact_path)
    return pd.read_csv(os.path.join(artifact_path, "output.csv"))
