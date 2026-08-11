import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# ---------- RANDOM FOREST ----------
def run_random_forest(df, target_col, dataset_name):
    print(f"\n===== Random Forest on {dataset_name} =====")

    if target_col not in df.columns:
        print("❌ Random Forest skipped (no label column found)")
        return None

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "model": "Random Forest",
        "dataset": dataset_name,
        "report": classification_report(y_test, y_pred, output_dict=True)
    }


# ---------- ISOLATION FOREST ----------
def run_isolation_forest(df, dataset_name):
    print(f"\n===== Isolation Forest on {dataset_name} =====")

    model = IsolationForest(contamination=0.05)
    preds = model.fit_predict(df)

    preds = pd.Series(preds).map({1: 0, -1: 1})

    anomaly_count = preds.sum()
    total = len(preds)

    print("\n📊 Anomaly Summary:")
    print(preds.value_counts())

    print(f"\n🔍 Anomaly Percentage: {(anomaly_count / total) * 100:.2f}%")

    return {
        "model": "Isolation Forest",
        "dataset": dataset_name,
        "anomalies": anomaly_count,
        "total": total
    }


# ---------- LOAD DATA ----------
network_df = pd.read_csv(r"C:\Users\goutham\Downloads\embedded_system_network_security_dataset.csv")
cloud_df = pd.read_csv(r"C:\Users\goutham\Downloads\Cloud_Anomaly_Dataset.csv")

# ---------- CLEAN ----------
network_df = network_df.select_dtypes(include=['number'])
cloud_df = cloud_df.select_dtypes(include=['number'])

# ---------- RUN ----------
results = []

print("\n================ NETWORK DATASET ================")
results.append(run_random_forest(network_df, "label", "Network"))
results.append(run_isolation_forest(network_df, "Network"))

print("\n================ CLOUD DATASET ================")
results.append(run_random_forest(cloud_df, "label", "Cloud"))  # will skip
results.append(run_isolation_forest(cloud_df, "Cloud"))


# ---------- FINAL COMPARISON ----------
print("\n================ FINAL COMPARISON ================")

for r in results:
    if r is None:
        continue

    if r["model"] == "Random Forest":
        accuracy = r["report"]["accuracy"]
        print(f"📌 {r['model']} on {r['dataset']} → Accuracy: {accuracy:.2f}")

    elif r["model"] == "Isolation Forest":
        anomaly_rate = (r["anomalies"] / r["total"]) * 100
        print(f"📌 {r['model']} on {r['dataset']} → Anomaly Rate: {anomaly_rate:.2f}%")

print("\n===== CLOUD DATASET =====")
run_random_forest(cloud_df, target_col="label")  
run_isolation_forest(cloud_df)
