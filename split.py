import random
from dataclasses import dataclass
import argparse

@dataclass
class SplitConfig:
	dataset: str
	ratio: float
	seed: int

def get_args() -> SplitConfig:
	parser = argparse.ArgumentParser(description="Split the dataset into training and validation sets")
	parser.add_argument("dataset", type=str, help="Path to the dataset")
	parser.add_argument("--ratio", type=float, default=0.8, help="Ratio of training data")
	parser.add_argument("--seed", type=int, default=-1, help="Seed for random number generator")
	args = parser.parse_args()
	return SplitConfig(dataset=args.dataset, ratio=args.ratio, seed=args.seed)


def split_config(config: SplitConfig) -> None:
	outputs: tuple[str, str] = ("train.csv", "validation.csv")
	lines: list[str] = []

	if config.seed != -1:
		random.seed(config.seed)
	with open(config.dataset, "r") as file:
		lines = [line.strip() for line in file if line.strip()]
	random.shuffle(lines)
	index = int(len(lines) * config.ratio)
	training: list[str] = lines[:index]
	validation: list[str] = lines[index:]
	with open(outputs[0], "w") as file:
		file.write("\n".join(training) + "\n")
	with open(outputs[1], "w") as file:
		file.write("\n".join(validation) + "\n")

if __name__ == "__main__":
	config = get_args()
	split_config(config)
	print(f"Split complete: train.csv ({config.ratio * 100:.0f}%) and validation.csv ({(1 - config.ratio) * 100:.0f}%)")
