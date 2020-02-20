import argparse
import os
import glob
import random

from tqdm import tqdm


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_input_dir", required=True, type=str, help="Root dataset input directory")
    parser.add_argument("--dataset_output_dir", required=True, type=str,
                        help="Root dataset output directory, with train, val[, test] subdir")
    parser.add_argument("--val_split", required=False, type=int, default=20, help="Validation split")
    parser.add_argument("--test_split", required=False, type=int, default=0, help="Test split")
    parser.add_argument("--balanced", action="store_true", help="Create balanced splits")
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    assert os.path.exists(args.dataset_input_dir), f"Dataset dir {args.dataset_input_dir} not found"

    assert os.path.exists(args.dataset_output_dir) is False, f"Output dir {args.dataset_output_dir} already exists"

    # Iterate over subclasses
    subdirs = next(os.walk(args.dataset_input_dir))[1]
    if len(subdirs) == 0:
        subdirs = [""]

    files = dict()

    for class_name in subdirs:
        extensions = ["bmp", "png", "jpg"]
        files_class = []
        for extension in extensions:
            files_class.extend(glob.glob(os.path.join(args.dataset_input_dir, class_name) + "/*." + extension))
        random.shuffle(files_class)
        files[class_name] = files_class

    print("Original distribution:")
    min_count = None
    for class_name, files_class in files.items():
        print(f"{class_name}: {len(files_class)}")
        if min_count is None or min_count > len(files_class):
            min_count = len(files_class)

    if args.balanced:
        print("Balancing dataset:")
        for class_name, files_class in files.items():
            files[class_name] = files_class[:min_count]
            print(f"{class_name}: {len(files_class)}")

    for class_name, files_class in files.items():
        train_files = files_class[:int(len(files_class) * (100 - args.val_split - args.test_split) / 100)]
        val_files = files_class[int(len(files_class) * (100 - args.val_split) / 100):
                                int(len(files_class) * (100 - args.val_split - args.test_split))]

        os.makedirs(os.path.join(args.dataset_output_dir, "train", class_name))
        os.makedirs(os.path.join(args.dataset_output_dir, "val", class_name))

        if args.test_split > 0:
            os.makedirs(os.path.join(args.dataset_output_dir, "test", class_name))

        print(f"Class {class_name} Train: {len(train_files)}")
        print(f"Class {class_name} Val: {len(val_files)}")

        test_files = None
        if args.test_split > 0:
            test_files = files_class[int(len(files_class) * (100 - args.test_split) / 100):]
            print(f"Class {class_name} Test: {len(test_files)}")


        for source in tqdm(train_files):
            try:
                os.symlink(source, os.path.join(args.dataset_output_dir, "train", class_name, os.path.basename(source)))
            except FileExistsError:
                pass

        for source in tqdm(val_files):
            try:
                os.symlink(source, os.path.join(args.dataset_output_dir, "val", class_name, os.path.basename(source)))
            except FileExistsError:
                pass

        if args.test_split > 0:
            for source in tqdm(test_files):
                try:
                    os.symlink(source,
                               os.path.join(args.dataset_output_dir, "test", class_name, os.path.basename(source)))
                except FileExistsError:
                    pass

    print("Success")


if __name__ == '__main__':
    main()
