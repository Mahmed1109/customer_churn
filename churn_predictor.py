import logging
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

DATA_PATH       = "customer_data.csv"
MODEL_PATH      = "churn_model.pkl"
TEST_SIZE       = 0.2
RANDOM_STATE    = 42
N_ESTIMATORS    = 100
DROP_COLUMNS    = ["customerID"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        log.info("Loaded %d rows and %d columns from '%s'.", *df.shape, path)
        return df
    except FileNotFoundError:
        log.error("File '%s' not found. Check DATA_PATH in the config.", path)
        sys.exit(1)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    null_counts = df.isnull().sum()
    if null_counts.any():
        log.warning(
            "Filling %d null value(s) across columns: %s",
            null_counts.sum(),
            null_counts[null_counts > 0].to_dict(),
        )
    df = df.fillna(0)

    if "Churn" not in df.columns:
        raise ValueError("Expected 'Churn' column not found in data.")

    y = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    if y.isnull().any():
        raise ValueError("'Churn' column contains unexpected values (expected 'Yes'/'No').")

    X = df.drop(columns=["Churn"])
    X = pd.get_dummies(X, drop_first=True)

    log.info(
        "Preprocessed data — %d features, %.1f%% churn rate.",
        X.shape[1],
        y.mean() * 100,
    )
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    log.info("Training RandomForestClassifier with %d estimators …", N_ESTIMATORS)
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    log.info("Training complete.")
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'─'*50}")
    print(f"  Model Accuracy : {accuracy * 100:.2f}%")
    print(f"{'─'*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Stay", "Actual Churn"],
        columns=["Pred Stay", "Pred Churn"],
    )
    print(cm_df)

    importances = (
        pd.Series(model.feature_importances_, index=X_test.columns)
        .nlargest(10)
        .round(4)
    )
    print("\nTop 10 Feature Importances:")
    for feat, score in importances.items():
        bar = "█" * int(score * 200)
        print(f"  {feat:<40} {score:.4f}  {bar}")
    print()


def show_sample_predictions(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    n: int = 5,
) -> None:
    sample = X_test.sample(n, random_state=RANDOM_STATE)
    preds  = model.predict(sample)
    print(f"\nSample predictions from test set ({n} rows):")
    for i, pred in enumerate(preds, 1):
        label = "CHURN" if pred == 1 else "STAY"
        print(f"  Sample {i}: {label}")


def _ask(prompt: str, valid: list[str] | None = None) -> str:
    while True:
        answer = input(prompt).strip()
        if not valid:
            return answer
        if answer.lower() in [v.lower() for v in valid]:
            return answer
        print(f"  Please enter one of: {', '.join(valid)}")


def _ask_int(prompt: str, valid: list[int] | None = None) -> int:
    while True:
        try:
            val = int(input(prompt).strip())
            if valid is None or val in valid:
                return val
            print(f"  Please enter one of: {valid}")
        except ValueError:
            print("  Invalid input — please enter a whole number.")


def _ask_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("  Invalid input — please enter a number (e.g. 65.50).")


def get_user_input() -> dict:
    print("\n--- Please enter customer details ---")
    yes_no        = ["Yes", "No"]
    yes_no_ns     = ["Yes", "No", "No internet service"]
    yes_no_np     = ["Yes", "No", "No phone service"]
    internet_opts = ["DSL", "Fiber optic", "No"]
    contract_opts = ["Month-to-month", "One year", "Two year"]
    payment_opts  = [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]

    return {
        "gender"          : _ask("Gender (Male/Female): ", ["Male", "Female"]),
        "SeniorCitizen"   : _ask_int("Senior Citizen (0=No, 1=Yes): ", [0, 1]),
        "Partner"         : _ask("Partner (Yes/No): ", yes_no),
        "Dependents"      : _ask("Dependents (Yes/No): ", yes_no),
        "tenure"          : _ask_int("Tenure in months: "),
        "PhoneService"    : _ask("Phone Service (Yes/No): ", yes_no),
        "MultipleLines"   : _ask("Multiple Lines (Yes/No/No phone service): ", yes_no_np),
        "InternetService" : _ask("Internet Service (DSL/Fiber optic/No): ", internet_opts),
        "OnlineSecurity"  : _ask("Online Security (Yes/No/No internet service): ", yes_no_ns),
        "OnlineBackup"    : _ask("Online Backup (Yes/No/No internet service): ", yes_no_ns),
        "DeviceProtection": _ask("Device Protection (Yes/No/No internet service): ", yes_no_ns),
        "TechSupport"     : _ask("Tech Support (Yes/No/No internet service): ", yes_no_ns),
        "StreamingTV"     : _ask("Streaming TV (Yes/No/No internet service): ", yes_no_ns),
        "StreamingMovies" : _ask("Streaming Movies (Yes/No/No internet service): ", yes_no_ns),
        "Contract"        : _ask("Contract (Month-to-month/One year/Two year): ", contract_opts),
        "PaperlessBilling": _ask("Paperless Billing (Yes/No): ", yes_no),
        "PaymentMethod"   : _ask(
            "Payment Method\n"
            "  1) Electronic check\n"
            "  2) Mailed check\n"
            "  3) Bank transfer (automatic)\n"
            "  4) Credit card (automatic)\n"
            "Your choice: ",
            payment_opts,
        ),
        "MonthlyCharges"  : _ask_float("Monthly Charges: "),
        "TotalCharges"    : _ask_float("Total Charges: "),
    }


def predict_new_customer(
    model: RandomForestClassifier,
    train_columns: pd.Index,
) -> None:
    choice = _ask("\nPredict churn for a new customer? (yes/no): ", ["yes", "no"])
    if choice.lower() != "yes":
        print("\nNo user-input prediction requested.")
        return

    input_data = get_user_input()
    user_df    = pd.DataFrame([input_data])
    user_df    = pd.get_dummies(user_df, drop_first=True)
    user_df    = user_df.reindex(columns=train_columns, fill_value=0)

    prediction  = model.predict(user_df)[0]
    probability = model.predict_proba(user_df)[0][1]

    print("\nPrediction Result:")
    if prediction == 1:
        print(f"  >> The customer is likely to CHURN  (confidence: {probability:.1%})")
    else:
        print(f"  >> The customer is likely to STAY   (confidence: {1 - probability:.1%})")


def main() -> None:
    raw_data = load_data(DATA_PATH)
    X, y     = preprocess(raw_data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    log.info("Train/test split — %d train rows, %d test rows.", len(X_train), len(X_test))

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)
    show_sample_predictions(model, X_test)

    joblib.dump(model, MODEL_PATH)
    log.info("Model saved to '%s'.", MODEL_PATH)

    predict_new_customer(model, X.columns)


if __name__ == "__main__":
    main()