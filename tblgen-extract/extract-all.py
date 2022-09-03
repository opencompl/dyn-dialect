from __future__ import annotations
import sys
import os
import subprocess

llvm_root = "llvm-project"
tblgen_extract_bin = "build/bin/tblgen-extract"


def generate_irdl_file(file: str,
                       output_folder: str,
                       print_command: bool = True) -> None:
    """
    Read a TableGen file and translate it to an IRDL file in the given folder
    if the TableGen file is correctly parsed.
    Also print the command used generate the file contents.
    """
    root, file = os.path.split(file)
    command = [
        f"./{tblgen_extract_bin}", f"{os.path.join(root, file)}",
        f"--I={llvm_root}/mlir/include", f"--I={root}"
    ]
    res = subprocess.run(command, capture_output=True)
    if res.returncode != 0:
        return None
    if (print_command):
        print(" ".join(command))

    output_file = open(os.path.join(output_folder, file), "w")
    output_file.write(res.stdout.decode())
    output_file.close()


def generate_all_irdl_files(output_folder: str) -> None:
    for root, _, files in os.walk(llvm_root + "/mlir/include"):
        for file in files:
            if file.endswith(".td"):
                generate_irdl_file(os.path.join(root, file), output_folder)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception(
            "Expected at least one command line argument: the "
            "path of the folder in which the IRDL files should be extracted.")
    files = generate_all_irdl_files(sys.argv[1])
