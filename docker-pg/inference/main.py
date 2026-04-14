import os
import tarfile

# if tarfile.is_tarfile("/input/input.tar"):
#     print("[inference-test] Input file is a tar file")
#     print("[inference-test] Unpacking input file...")
#     with tarfile.open("/input/input.tar") as tar:
#         tar.extractall("/input")

for root, dirs, files in os.walk("/input"):
    for name in files:
        print(os.path.join(root, name))

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
    print(e)

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

print("[inference-test] Packing output file...")
with tarfile.open("/output/output.tar", "w") as tar:
    tar.add("/output/output.txt", arcname="output.txt")

