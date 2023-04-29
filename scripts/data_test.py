from guided_diffusion.image_datasets import _list_image_files_recursively
import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", type=str, required=True, help="The data directory."
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    all_files = _list_image_files_recursively(args.data_dir)
    print("amount of files:", len(all_files))
    print(all_files[:10])
