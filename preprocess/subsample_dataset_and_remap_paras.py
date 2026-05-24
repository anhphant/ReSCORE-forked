import argparse
import random
import os

from preprocess.lib import (
    read_jsonl,
    write_jsonl
)

random.seed(13370)


def get_instance_id(instance):

    if "question_id" in instance:
        return instance["question_id"]

    if "_id" in instance:
        return instance["_id"]

    if "id" in instance:
        return instance["id"]

    raise ValueError(
        f"Cannot find id field: {instance.keys()}"
    )


def main(args):

    avoid_question_ids_file_path = None

    if args.set_name == "test":

        dev_file_path = os.path.join(
            "data",
            "processed_data",
            args.dataset_name,
            "dev_subsampled.jsonl"
        )

        if os.path.exists(dev_file_path):

            avoid_question_ids_file_path = \
                dev_file_path

        sample_size = 500

    elif args.set_name == "dev_diff_size":

        avoid_question_ids_file_path = \
            os.path.join(
                "data",
                "processed_data",
                args.dataset_name,
                "test_subsampled.jsonl"
            )

        sample_size = args.sample_size

    else:

        sample_size = 100

    # ===== FIX BUG CHÍNH =====
    input_split = args.set_name

    if args.set_name == "dev_diff_size":
        input_split = "dev"

    input_file_path = os.path.join(
        "data",
        "processed_data",
        args.dataset_name,
        f"{input_split}.jsonl"
    )

    print(
        "Loading:",
        input_file_path
    )

    instances = read_jsonl(
        input_file_path
    )

    print(
        "Raw instances:",
        len(instances)
    )

    if len(instances):

        print(
            "Example keys:",
            instances[0].keys()
        )

    if avoid_question_ids_file_path:

        print(
            "Avoid file:",
            avoid_question_ids_file_path
        )

        avoid_instances = read_jsonl(
            avoid_question_ids_file_path
        )

        avoid_ids = set(

            get_instance_id(x)

            for x in avoid_instances

        )

        before = len(instances)

        instances = [

            x

            for x in instances

            if get_instance_id(x)
            not in avoid_ids

        ]

        print(
            "Filtered:",
            before,
            "->",
            len(instances)
        )

    if len(instances) == 0:

        raise ValueError(
            f"No remaining samples "
            f"for {args.dataset_name} "
            f"{args.set_name}"
        )

    sample_size = min(
        sample_size,
        len(instances)
    )

    instances = random.sample(
        instances,
        sample_size
    )

    if args.set_name == "dev_diff_size":

        output_file_path = os.path.join(

            "data",
            "processed_data",
            args.dataset_name,

            f"dev_{args.sample_size}_subsampled.jsonl"
        )

    else:

        output_file_path = os.path.join(

            "data",
            "processed_data",
            args.dataset_name,

            f"{args.set_name}_subsampled.jsonl"
        )

    write_jsonl(
        instances,
        output_file_path
    )

    print(
        f"Saved {len(instances)} "
        f"instances -> "
        f"{output_file_path}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--dataset_name",

        choices=(
            "hotpotqa",
            "2wikimultihopqa",
            "musique",
            "vimqa",
            "nq",
            "trivia",
            "squad"
        ),

        required=True
    )

    parser.add_argument(

        "--set_name",

        choices=(
            "dev",
            "test",
            "dev_diff_size"
        ),

        required=True
    )

    parser.add_argument(

        "--sample_size",

        type=int,

        default=500
    )

    args = parser.parse_args()

    main(args)