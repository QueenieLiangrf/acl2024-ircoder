import argparse
import os
import pandas as pd
import pickle
import subprocess
from multiprocessing import Pool, current_process
from tqdm import tqdm
import shutil

# Define the paths as required
root_path = "/kaggle/working/input/Go"  # Adjust this if needed
out_path = "/kaggle/working/output/Go"  # Adjust this if needed

def compile_worker(dir_list_chunk):
    result = pd.DataFrame(columns=["Source", "Perf_Compile_Status", "Size_Compile_Status"])
    result = result.set_index("Source")

    # Ensure the Logs directory exists
    logs_dir = os.path.join(out_path, "Logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create log file paths using the process ID
    out_log_path = os.path.join(logs_dir, f"out_{current_process().pid}.txt")
    err_log_path = os.path.join(logs_dir, f"err_{current_process().pid}.txt")

    # Open log files in append mode
    with open(out_log_path, "a+") as outfile, open(err_log_path, "a+") as errfile:
        for folder in tqdm(dir_list_chunk, desc=f"Worker - {current_process().pid}: "):
            src_path = os.path.join(root_path, folder, "source.go")
            dest_source_path = os.path.join(out_path, folder, "source.go")
            perf_path = os.path.join(out_path, folder, "llvm_O3.ll")
            size_path = os.path.join(out_path, folder, "llvm_OZ.ll")

            # Ensure output directories exist
            os.makedirs(os.path.dirname(perf_path), exist_ok=True)
            os.makedirs(os.path.dirname(size_path), exist_ok=True)

            # Copy the source file to the output directory
            try:
                shutil.copy(src_path, dest_source_path)
            except Exception as e:
                print(f"Error copying source file {src_path} to {dest_source_path}: {e}")
                result.loc[folder] = [-1, -1]
                continue

            try:
                # Compile to LLVM IR for performance optimization (-O3)
                perf_returncode = subprocess.run(
                    [
                        "llvm-goc", "-S",
                        "-O3",
                        "-fno-inline",
                        "-fno-integrated-as",
                        "-emit-llvm",
                        "-fno-debug-info-for-profiling",
                        "-fno-go-debug-optimization",
                        src_path,
                        "-o", perf_path
                    ],
                    stdout=outfile,
                    stderr=errfile,
                    check=False  # Set to True if you want to raise an error on non-zero return code
                ).returncode
            except Exception as e:
                print(f"Error compiling {src_path} for performance optimization: {e}")
                perf_returncode = -1  # Indicate failure

            try:
                # Compile to LLVM IR for size optimization (-Oz)
                size_returncode = subprocess.run(
                    [
                        "llvm-goc", "-S",
                        "-Os",
                        "-fno-inline",
                        "-fno-integrated-as",
                        "-emit-llvm",
                        "-fno-debug-info-for-profiling",
                        "-fno-go-debug-optimization",
                        src_path,
                        "-o", size_path
                    ],
                    stdout=outfile,
                    stderr=errfile,
                    check=False  # Set to True if you want to raise an error on non-zero return code
                ).returncode
            except Exception as e:
                print(f"Error compiling {src_path} for size optimization: {e}")
                size_returncode = -1  # Indicate failure

            result.loc[folder] = [perf_returncode, size_returncode]

    return result

def main(args):
    dir_list = os.listdir(root_path)
    dir_list.sort()
    if args.subset:
        dir_list = dir_list[:args.subset]
    print(f"Obtained {len(dir_list)} source files for compilation")
    
    os.makedirs(out_path, exist_ok=True)  # Ensure the output directory exists

    pickle_path = os.path.join(out_path, "dir_list.pickle")
    with open(pickle_path, "wb") as sp:
        pickle.dump(dir_list, sp)

    dir_list_chunked = [dir_list[i::args.num_workers] for i in range(args.num_workers)]
    with Pool(args.num_workers) as pool:
        results_list = pool.map(compile_worker, dir_list_chunked)
        results_df = pd.concat(results_list, ignore_index=False)
        results_df.to_parquet(os.path.join(out_path, "results.parquet"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=8,
        help="Number of workers to spawn for compilation"
    )
    parser.add_argument(
        "--subset",
        type=int,
        required=False,
        default=None,
        help="Number of source files to compile"
    )
    args = parser.parse_args()
    main(args)
