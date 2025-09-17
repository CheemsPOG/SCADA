import argparse
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

# Purpose:
# - Load the saved preprocessing pipeline and model
# - Predict power for either: (a) a built-in sample row, or (b) rows from a CSV
# - Keep it simple, self-contained, and robust to working directory


BASE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = BASE_DIR / "one_hot_pipeline.pkl"
MODEL_PATH = BASE_DIR / "one_hot_model.pkl"


def robust_load(pkl_path: Path):
    """
    Load a pickle with joblib; if that fails due to notebook-defined classes,
    fall back to cloudpickle for better compatibility.
    """
    try:
        return joblib.load(pkl_path)
    except AttributeError as err:
        try:
            import cloudpickle  # type: ignore
        except Exception:
            raise RuntimeError(
                "Could not load artifacts due to custom classes. "
                "Install cloudpickle and retry: pip install cloudpickle. "
                f"Original error: {err}"
            ) from err
        with open(pkl_path, "rb") as f:
            return cloudpickle.load(f)


def load_artifacts():
    """Load pipeline and model from disk, with clear errors if missing."""
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Missing pipeline file: {PIPELINE_PATH}. Place it next to test.py."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}. Place it next to test.py."
        )
    pipeline = robust_load(PIPELINE_PATH)
    model = robust_load(MODEL_PATH)
    return pipeline, model


def load_input_df(csv_path: Optional[str]) -> pd.DataFrame:
    """
    If csv_path is provided, read it; otherwise return a single sample row.
    The columns must match what the pipeline expects.
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        return df

    # Example SCADA row (adjust values as needed)
    sample_data = {
        "Date/Time": ["31 12 2018 23:10"],
        "Wind Speed (m/s)": [11.404030],
        "Theoretical_Power_Curve (KWh)": [3397.190793],
        "Wind Direction (°)": [80.502724],
    }
    return pd.DataFrame(sample_data)


def predict(df: pd.DataFrame, pipeline, model) -> pd.Series:
    """Transform inputs via pipeline, then predict with the model."""
    transformed = pipeline.transform(df)
    preds = model.predict(transformed)
    return pd.Series(preds)


def main():
    parser = argparse.ArgumentParser(
        description="Test the wind turbine power prediction model"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to a CSV with input rows to score",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Print only the first N predictions when using CSV",
    )
    args = parser.parse_args()

    pipeline, model = load_artifacts()
    df_in = load_input_df(args.csv)
    preds = predict(df_in, pipeline, model)

    if args.csv:
        print(f"Total rows scored: {len(preds)}")
        print("First predictions:")
        print(preds.head(args.n).to_string(index=False))
    else:
        print(f"Predicted Power (kW): {preds.iloc[0]}")


if __name__ == "__main__":
    main()




