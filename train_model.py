from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier


def main() -> None:
	features = [
		[25000, 5000, 22],
		[40000, 12000, 29],
		[55000, 15000, 35],
		[32000, 9000, 27],
		[78000, 20000, 41],
		[90000, 25000, 50],
		[120000, 30000, 46],
		[28000, 7000, 24],
	]
	risk = [1, 1, 0, 1, 0, 0, 0, 1]

	model = RandomForestClassifier(n_estimators=100, random_state=42)
	model.fit(features, risk)

	output_path = Path("model.joblib")
	joblib.dump(model, output_path)

	print("Model ready!")


if __name__ == "__main__":
	main()
