import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report

# ---------- FUNCTION: Random Forest ----------
def run_random_forest(df, target_col):
    print("\n--- Random Forest ---")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))


# ---------- FUNCTION: Isolation Forest ----------
def run_isolation_forest(df):
    print("\n--- Isolation Forest ---")

    model = IsolationForest(contamination=0.05)
    preds = model.fit_predict(df)

    preds = pd.Series(preds).map({1: 0, -1: 1})

    print("Anomalies detected:")
    print(preds.value_counts())


# ---------- LOAD DATASETS ----------
network_df = pd.read_csv("network_dataset.csv")
cloud_df = pd.read_csv("cloud_dataset.csv")


# ---------- CLEAN DATA ----------

network_df = network_df.select_dtypes(include=['number'])
cloud_df = cloud_df.select_dtypes(include=['number'])


# ---------- RUN MODELS ----------

print("===== NETWORK DATASET =====")
run_random_forest(network_df, target_col="label")  
run_isolation_forest(network_df)

print("\n===== CLOUD DATASET =====")
run_random_forest(cloud_df, target_col="label")  
run_isolation_forest(cloud_df)
