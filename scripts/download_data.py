import os
import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download spect dataset from Hugging Face")
    parser.add_argument("--repo_id", type=str, default="TheGreatPatric/spect",
                        help="Hugging Face dataset repo ID")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to download")
    parser.add_argument("--output_dir", type=str, default="data_split_cropped",
                        help="Output directory for downloaded data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset '{args.repo_id}' split='{args.split}'...")
    dataset = load_dataset(args.repo_id, split=args.split, trust_remote_code=True)

    print(f"Saving {len(dataset)} samples to '{args.output_dir}/'...")
    dataset.save_to_disk(args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
