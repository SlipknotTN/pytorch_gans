import argparse
import os

import cv2
import glob
from tqdm import tqdm


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Prepare male/female directory from LFW dataset")
    parser.add_argument("--dataset_root_dir", required=True, type=str,
                        help="Dataset root dir")
    parser.add_argument("--dataset_root_output_dir", required=True, type=str,
                        help="Output directory that will the same source subdirectories")
    parser.add_argument("--height", required=True, type=int, help="Resize height")
    parser.add_argument("--width", required=True, type=int, help="Resize width")
    parser.add_argument("--bw", action="store_true", help="B/W color")
    args = parser.parse_args()
    return args


def main():
    # Load params
    args = do_parsing()
    print(args)

    walker = os.walk(args.dataset_root_dir)
    root, dirs, _ = next(walker)

    for dir in tqdm(dirs):
        os.makedirs(os.path.join(args.dataset_root_output_dir, dir), exist_ok=False)
        images = glob.glob(os.path.join(args.dataset_root_dir, dir) + "/*.jpg")
        for image_file in tqdm(images, desc=f"{dir} images"):
            if args.bw:
                image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(image_file, cv2.IMREAD_COLOR)
            image_rsz = cv2.resize(image, (args.width, args.height))
            dst_filepath = os.path.join(args.dataset_root_output_dir, dir, os.path.basename(image_file))
            cv2.imwrite(dst_filepath, image_rsz)


if __name__ == "__main__":
    main()
