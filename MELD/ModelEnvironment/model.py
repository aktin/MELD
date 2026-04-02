import io
import os
import subprocess

import pandas as pd


def run_inference(input_data: pd.DataFrame, artifact_path: str = "artifact"):
    model_py_path = os.path.join("..", artifact_path, ".venv", "bin", "python")
    proc = subprocess.Popen([model_py_path, "inference.py"],
                            cwd=artifact_path,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE,
                            text=True,
                            env={"TF_CPP_MIN_LOG_LEVEL": "3",
                                 "TF_ENABLE_ONEDNN_OPTS": "0"})
    input_str = input_data.to_csv(index=False)
    stdout, stderr = proc.communicate(input=input_str)

    if proc.returncode != 0:
        raise RuntimeError(f"Inference failed:\n{stderr}")

    return pd.read_csv(io.StringIO(stdout))
