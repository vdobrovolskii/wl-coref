#!/usr/bin/env python3

import argparse
import os
import re
import subprocess


def extract_f1(proc: subprocess.CompletedProcess) -> float:
    prev_line = ""
    curr_line = ""
    for line in str(proc.stdout).splitlines():
        prev_line = curr_line
        curr_line = line
    return float(re.search(r"F1:\s*([0-9.]+)%", prev_line).group(1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get conll score")
    parser.add_argument("section", help="The name of the experiment.")
    parser.add_argument("data_split", choices=("train", "dev", "test"))
    parser.add_argument("epoch", type=int)
    parser.add_argument("--log-dir", default="data/conll_logs")
    args = parser.parse_args()

    filename_prefix = f"{args.section}_{args.data_split}_e{args.epoch}"

    gold = os.path.join(args.log_dir, f"{filename_prefix}.gold.conll")
    pred = os.path.join(args.log_dir, f"{filename_prefix}.pred.conll")

    part_a = ["perl", "reference-coreference-scorers/scorer.pl"]
    part_b = [gold, pred]
    kwargs = {"capture_output": True, "check": True, "text": True}

    results = []
    for metric in "muc", "ceafe", "bcub":
        results.append(
            extract_f1(subprocess.run(part_a + [metric] + part_b, **kwargs)))
        print(metric, results[-1])

    print("avg", sum(results) / len(results))
