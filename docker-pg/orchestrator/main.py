import io
import os
import shutil
import subprocess
import tarfile

import docker
from docker.models.containers import Container
from docker.models.volumes import Volume

INFERENCE_IMAGE = 'inference-test'
ORCHESTRATOR_IMAGE = 'orchestrator'

client = docker.from_env()


def create_input_volume():
    print("[orchestrator-test] Creating input volume...")
    return client.volumes.create('input')


def create_output_volume():
    print("[orchestrator-test] Creating output volume...")
    return client.volumes.create('output')


def write_input_file_in_volume():
    print("[orchestrator-test] Writing input file in volume...")
    result = client.containers.run("alpine", volumes={'input': {'bind': '/input', 'mode': 'rw'}},
                                   command=["sh", "-c", "echo 'Test input' > /input/input.txt"], remove=True,
                                   detach=False)
    print(result.decode('utf-8'))


def read_output_file_in_volume():
    print("[orchestrator-test] Reading output file in volume...")
    result = client.containers.run("alpine", volumes={'output': {'bind': '/output', 'mode': 'ro'}},
                                   command=["sh", "-c", "cat /output/output.txt"], remove=True, stdout=True,
                                   stderr=True, detach=False)
    print(result.decode('utf-8'))


def start_inference(container: Container):
    print("[orchestrator-test] Starting inference...")
    container.start()
    container.wait()
    logs = container.logs(stdout=True, stderr=True)
    container.stop()
    print(logs.decode('utf-8'))


def remove_inference_container(container: Container):
    print("[orchestrator-test] Removing inference container...")
    container.remove()


def create_inference_container(volumes=None):
    print("[orchestrator-test] Creating inference container...")
    return client.containers.create(INFERENCE_IMAGE, detach=True, volumes=volumes, privileged=False, cap_drop="all")


def run_inference(volumes=None):
    container = create_inference_container(volumes=volumes)
    start_inference(container)
    remove_inference_container(container)


def cleanup_volumes(input_volume: Volume, output_volume: Volume):
    print("[orchestrator-test] Cleaning up volumes...")
    input_volume.remove()
    output_volume.remove()


def test_volumes():
    print("[orchestrator-test] ************************************************\nTesting volumes...")
    input_volume = create_input_volume()
    output_volume = create_output_volume()
    write_input_file_in_volume()
    run_inference(volumes={'input': {'bind': '/input', 'mode': 'ro'}, 'output': {'bind': '/output', 'mode': 'rw'}})
    read_output_file_in_volume()
    cleanup_volumes(input_volume, output_volume)


def create_input_mount_folder():
    print("[orchestrator-test] Creating input mountpoint...")
    if not os.path.exists("/shared/input"):
        os.makedirs("/shared/input")


def create_output_mount_folder():
    print("[orchestrator-test] Creating output mountpoint...")
    if not os.path.exists("/shared/output"):
        os.makedirs("/shared/output")


def write_input_file_in_mount():
    print("[orchestrator-test] Writing input file in mount...")
    # set_folder_to_write("/shared/input")
    with open("/shared/input/input.txt", "w") as f:
        f.write("Test input")


def read_output_file_in_mount():
    print("[orchestrator-test] Reading output file in mount...")
    with open("/shared/output/output.txt", "r") as f:
        print(f.read())


def cleanup_mounts():
    set_folder_to_write("/shared/input")
    print("[orchestrator-test] Cleaning up mounts...")
    shutil.rmtree("/shared/input")
    shutil.rmtree("/shared/output")


def test_mounts():
    print("[orchestrator-test] ************************************************\nTesting bind mounts...")
    create_input_mount_folder()
    create_output_mount_folder()
    write_input_file_in_mount()
    run_inference(volumes={
        "/home/shuening/code/KlimaNot/docker-pg/orchestrator/shared/input": {"bind": "/input", "mode": "ro"},
        "/home/shuening/code/KlimaNot/docker-pg/orchestrator/shared/output": {"bind": "/output", "mode": "rw"},
    })
    read_output_file_in_mount()
    cleanup_mounts()


def set_folder_to_write(path: str):
    print(f"[orchestrator-test] Setting folder {path} to write...")
    for root, dirs, files in os.walk(path):
        # Make current directory read+execute only (no write)
        os.chmod(root, 0o777)

        # Make subdirectories read+execute only
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)

        # Make files read-only
        for f in files:
            os.chmod(os.path.join(root, f), 0o777)


def set_folder_to_read_only(path: str):
    print(f"[orchestrator-test] Setting folder {path} to read-only...")
    for root, dirs, files in os.walk(path):
        # Make current directory read+execute only (no write)
        os.chmod(root, 0o555)

        # Make subdirectories read+execute only
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o555)

        # Make files read-only
        for f in files:
            os.chmod(os.path.join(root, f), 0o444)


def copy_input_to_inference(container: Container):
    print("[orchestrator-test] Copying input to inference container...")
    subprocess.run(["docker", "cp", "/shared/input/", f"{container.id}:/"])


def copy_output_file(container: Container):
    print("[orchestrator-test] Copying output file from inference container...")
    subprocess.run(["docker", "cp", f"{container.id}:/output/", "/shared/"])


def test_docker_copy():
    print("[orchestrator-test] ************************************************\nTesting docker copy...")
    create_input_mount_folder()
    create_output_mount_folder()
    write_input_file_in_mount()
    set_folder_to_read_only("/shared/input")
    container = create_inference_container()
    copy_input_to_inference(container)
    start_inference(container)
    copy_output_file(container)
    cleanup_mounts()


def copy_input_data(container: Container):
    buf = io.BytesIO()

    print("[orchestrator-test] Archiving input data...")
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add("/shared/input", arcname="")

    buf.seek(0)

    print("[orchestrator-test] Copying input data to inference container...")
    container.put_archive("/input", buf.getvalue())


def unpacking_output_data(container):
    print("[orchestrator-test] Copying output data from inference container...")
    bits, _ = container.get_archive("/output/output.tar")

    print("[orchestrator-test] Unpacking output data...")
    with open("/shared/output/output.tar", "wb") as f:
        for chunk in bits:
            f.write(chunk)

def test_docker_copy_and_unpack():
    print("[orchestrator-test] ************************************************\nStarting docker api test...")
    create_input_mount_folder()
    create_output_mount_folder()
    write_input_file_in_mount()
    container = create_inference_container()
    copy_input_data(container)
    start_inference(container)
    unpacking_output_data(container)
    container.remove()
    cleanup_mounts()

if __name__ == '__main__':
    test_docker_copy_and_unpack()