import os

try:
    print("[inference-test] Reading input file...")
    with open("/input/input.txt", "r") as f:
        input_data = f.read()
        print(f"[inference-test] Input file content: {repr(input_data)}")
        if input_data.startswith("Test input"):
            print("[inference-test] Did work: pass")
        else:
            print("[inference-test] Did not work: fail")
except Exception as e:
    print("[inference-test] Did not work: fail")

try:
    print("[inference-test] Writing input file...")
    with open("/input/input.txt", "w") as f:
        f.write("Should not work")
        print("[inference-test] Did work: fail")
except Exception as e:
    print("[inference-test] Did not work: pass")

try:
    print("[inference-test] Trying to change permissions of input file...")
    os.chmod("/input/input.txt", 0o777)
    print("[inference-test] Did work: fail")
except Exception as e:
    print("[inference-test] Did not work: pass")

try:
    print("[inference-test] Writing output file...")
    with open("/output/output.txt", "w") as f:
        f.write("Test output")
        print("[inference-test] Did work: pass")
    with open("/output/output.txt", "r") as f:
        output_data = f.read()
except Exception as e:
    print("[inference-test] Did not work: fail")