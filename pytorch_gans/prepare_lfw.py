import argparse
import csv
import os
from shutil import copy


from tqdm import tqdm


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Prepare male/female directory from LFW dataset")
    parser.add_argument("--dataset_root_dir", required=True, type=str,
                        help="Dataset root dir, must contain lfw_attributes.txt")
    parser.add_argument("--full_dataset_subdir", required=False, default="full", type=str,
                        help="Subdir containing LFW original dataset")
    parser.add_argument("--dataset_root_output_dir", required=True, type=str,
                        help="Output directory that will contain male and female subdirectories")
    parser.add_argument("--min_confidence", required=False, type=float, default=0.5,
                        help="Minimum value to accept male/female classication")
    args = parser.parse_args()
    return args


def main():

    # Load params
    args = do_parsing()
    print(args)

    os.makedirs(os.path.join(args.dataset_root_output_dir, "female"), exist_ok=False)
    os.makedirs(os.path.join(args.dataset_root_output_dir, "male"), exist_ok=False)

    with open(os.path.join(args.dataset_root_dir, "lfw_attributes.txt")) as fp:
        csv_reader = csv.reader(fp, delimiter="\t")
        next(csv_reader)
        header = next(csv_reader)[1:]
        for row in tqdm(csv_reader):
            print(f"{row[0]}: {row[header.index('Male')]}")
            male_prediction = float(row[header.index('Male')])
            abs_value = abs(male_prediction)
            if abs_value >= args.min_confidence:
                print("Accepted")
                image_number = row[header.index('imagenum')]
                person = row[header.index('person')]
                person = person.replace(" ", "_")
                basename = person + "_" + str(image_number).zfill(4) + ".jpg"
                classname = "male" if male_prediction >= 0 else "female"
                src_filename = os.path.join(args.dataset_root_dir, args.full_dataset_subdir, person, basename)
                assert os.path.exists(src_filename), f"Src file {src_filename} not exists"
                dst_filename = os.path.join(args.dataset_root_output_dir, classname, basename)
                copy(src_filename, dst_filename)
            else:
                print("Discarded")


if __name__ == "__main__":
    main()
