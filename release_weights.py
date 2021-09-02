""" Takes a saved model and re-saves it without optimizer and scheduler weights
to save space """

import argparse
import os

import torch


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("path", help="Path to saved model weights")
    args = argparser.parse_args()

    state_dict = torch.load(args.path)
    for key in ("general_optimizer", "general_scheduler",
                "bert_optimizer", "bert_scheduler"):
        state_dict.pop(key, None)

    dirname = os.path.dirname(args.path)
    filename = os.path.basename(args.path)
    filename, extension = os.path.splitext(filename)
    result_filename = f"{filename}_release{extension}"
    result_path = os.path.join(dirname, result_filename)

    print(f"Writing result to {result_path}")

    torch.save(state_dict, result_path)
