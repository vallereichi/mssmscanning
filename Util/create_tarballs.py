import os
import tarfile
import argparse



def create_tarball(source_dir: str, output_file: str) -> None:
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def compress_single_folder(dir_path: str, output_dir:str | None = None) -> None:
    output_dir = f"{dir_path}" if output_dir is None else output_dir
    base_name = os.path.basename(dir_path.rstrip(os.sep))
    output_file = os.path.join(output_dir, f"{base_name}.tar.gz")
    create_tarball(dir_path, output_file)
    print(f"compressed {output_file}")

def compress_all_subfolders(parent_dir: str, output_dir: str | None = None) -> None:
    output_dir = os.path.join(parent_dir, "compressed") if output_dir is None else output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for target_dir in os.listdir(parent_dir):
        source_dir = os.path.join(parent_dir, target_dir)
        if os.path.isdir(source_dir) and target_dir != "compressed":
            output_file = os.path.join(output_dir, f"{target_dir}.tar.gz")
            create_tarball(source_dir, output_file)
            print(f"compressed {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Compress one or multiple folders into a tarball")
    parser.add_argument("path", help="path to directory")
    parser.add_argument("--out", default=None, help="specify an output directory")
    parser.add_argument("--all", action="store_true", help="if flag is set, all subfolders of the specified path will be compressed into seperate tarballs")


    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise ValueError(f"path does not exist: {args.path}")

    if args.all:
        compress_all_subfolders(args.path, args.out)
    else:
        compress_single_folder(args.path, args.out)

if __name__ == "__main__":
    main()
