import random
from dataclasses import dataclass

@dataclass
class SplitConfig:
	dataset: str
	ratio: float
	seed: int

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
		file.write("\n".join(training))
	with open(outputs[1], "w") as file:
		file.write("\n".join(validation))