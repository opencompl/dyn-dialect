from __future__ import annotations
import sys
import os
import subprocess
import argparse

mlir_root = "llvm-project/mlir"
tblgen_extract_bin = "build/bin/tblgen-extract"


def generate_irdl_file(file: str,
                       root_folder: str,
                       output_folder: str,
                       args: str = "",
                       print_failing_command: bool = False) -> None:
    """
    Read a TableGen file and translate it to an IRDL file in the given folder
    if the TableGen file is correctly parsed.
    Also print the command used generate the file contents.
    """
    root, file = os.path.split(file)
    command = [
        f"./{tblgen_extract_bin}", f"{os.path.join(root, file)}",
        f"--I={mlir_root}/include", f"--I={root}", f"--I={root_folder}"
    ]
    if args != "":
        command.append(args)
    res = subprocess.run(command, capture_output=True)
    if res.returncode != 0:
        if (print_failing_command):
            print()
            print(f"Failed to generate IRDL file for {file}:")
            print(" ".join(command))
            print(res.stderr.decode("utf-8"))
            print()
        return None
    print(" ".join(command))

    output_file = open(os.path.join(output_folder, file), "w")
    output_file.write(res.stdout.decode())
    output_file.close()


def generate_all_irdl_files(project_folder: str,
                            output_folder: str,
                            args: str,
                            print_failing_command: bool = False) -> None:
    for root, _, files in os.walk(project_folder):
        for file in files:
            if file.endswith(".td"):
                generate_irdl_file(os.path.join(root, file),
                                   project_folder,
                                   output_folder,
                                   args=args,
                                   print_failing_command=print_failing_command)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Tablegen to IRDL extractor")

    arg_parser.add_argument(
        "-o",
        type=str,
        required=True,
        help="Folder where to store the generated IRDL files")

    arg_parser.add_argument("-d",
                            type=str,
                            default=mlir_root + "/include",
                            help="Folder where the TableGen files are located")

    arg_parser.add_argument("--args",
                            type=str,
                            default="",
                            help="Arguments to pass to tblgen-extract")

    arg_parser.add_argument(
        "--print-failing-command",
        action="store_true",
        help="Print the command used to generate the IRDL file "
        "if it fails, and its error message.")

    args = arg_parser.parse_args()

    output_folder = args.o
    directory = args.d

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    generate_all_irdl_files(directory, output_folder, args.args,
                            args.print_failing_command)
