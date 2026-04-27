from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from feature_engineering import add_feature_engineering


def main() -> None:
	training_rows = [
		[25000, 5000, 22, "Agriculture", 12000, 4000, 1],
		[40000, 12000, 29, "Salary", 18000, 5500, 1],
		[55000, 15000, 35, "Business", 22000, 7000, 0],
		[32000, 9000, 27, "Agriculture", 15000, 6000, 1],
		[78000, 20000, 41, "Salary", 25000, 9000, 0],
		[90000, 25000, 50, "Business", 30000, 11000, 0],
		[120000, 30000, 46, "Remittance", 35000, 14000, 0],
		[28000, 7000, 24, "Agriculture", 13000, 4500, 1],
	]
	training_frame = pd.DataFrame(
		training_rows,
		columns=[
			"Monthly_Income",
			"Loan_Amount",
			"Age",
			"Primary_Income_Source",
			"Essential_Expenses",
			"Monthly_Installment",
			"Risk",
		],
	)
	engineered_frame = add_feature_engineering(training_frame)
	engineered_frame["Debt_Service_Coverage"] = (
		engineered_frame["Stress_Tested_Income"] - engineered_frame["Essential_Expenses"]
	) / engineered_frame["Monthly_Installment"].replace(0, pd.NA)

	features = engineered_frame[["Monthly_Income", "Loan_Amount", "Age"]]
	risk = engineered_frame["Risk"]

	model = RandomForestClassifier(n_estimators=100, random_state=42)
	model.fit(features, risk)

	output_path = Path("model.joblib")
	joblib.dump(model, output_path)

	reports_dir = Path("reports")
	reports_dir.mkdir(exist_ok=True)
	engineered_frame.to_csv(reports_dir / "training_engineered_features.csv", index=False)

	print("Model ready!")


if __name__ == "__main__":
	main()
