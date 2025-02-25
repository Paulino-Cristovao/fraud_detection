import os
import pandas as pd
from train import load_data, preprocess_data

def test_load_and_preprocess():
    # Create a small temporary CSV file
    test_csv = "data/test.csv"
    os.makedirs("data", exist_ok=True)
    df_test = pd.DataFrame({
        "V1": [0.1, 0.2, 0.3],
        "V2": [0.2, 0.3, 0.4],
        "Class": [0, 1, 0]
    })
    df_test.to_csv(test_csv, index=False)
    df_loaded = load_data(test_csv)
    X, y = preprocess_data(df_loaded)
    # Expect 2 features and 3 samples
    assert X.shape == (3, 2)
    assert y.shape == (3,)
    os.remove(test_csv)
