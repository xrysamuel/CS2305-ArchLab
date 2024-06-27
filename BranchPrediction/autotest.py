#!/usr/bin/env python3

import os
import threading
import re

def extract_second_field(string):
    fields = string.split()
    if len(fields) >= 2:
        return fields[1]
    else:
        return None

gabps_path = "gapbs"

def log_stat(bp, workload):
    name = bp + "-" + workload

    with open("log/log.txt", 'a') as log_file:
        log_file.write(name + "\t")
        with open(f"log/m5out-{name}/stats.txt", 'r') as stats_file:
            for line in stats_file:
                if "condPredictedTaken" in line:
                    pass
                elif "condPredicted" in line:
                    number = extract_second_field(line)
                    log_file.write(number + "\t")
                elif "condIncorrect" in line:
                    number = extract_second_field(line)
                    log_file.write(number + "\t")
                elif "cpu.ipc" in line:
                    number = extract_second_field(line)
                    log_file.write(number + "\t")
        log_file.write("\n")


def run(bp, workload):
    name = bp + "-" + workload

    os.environ["BP"] = bp
    os.environ["WORKLOAD"] = workload
    os.environ["GAPBS_PATH"] = gabps_path
    os.system(f"gem5/build/X86/gem5.opt --outdir=log/m5out-{name} BPTest/bp.py")


if not os.path.exists("log"):
    os.mkdir("log")

bps = ["LocalBP", "TournamentBP", "TAGE", "LocalBP1Bit", "LocalBP3Bit"]
workloads = ["bfs", "bc", "cc"]
threads = []

for bp in bps:
    for workload in workloads:
        thread = threading.Thread(target=run, args=(bp, workload))
        thread.start()
        threads.append(thread)

for thread in threads:
    thread.join()

for bp in bps:
    for workload in workloads:
        log_stat(bp, workload)