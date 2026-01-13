
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_curve, auc, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# ------------------------------
# 1) Load datasets
# ------------------------------
paths = {
    "longdmbl": "imu_smooth_m2.csv",
    "underdmbl": "imu_smooth_m3.csv",
    "rotationdmbl": "imu_smooth_m4.csv"
}

colnames = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch", "yaw"]
channels = colnames[1:]  # exclude timestamp

# Read and store
raw_data = {}
for label, p in paths.items():
    df = pd.read_csv(p, header=None)
    # First row seems to have column names incorrectly, so enforce ours
    df.columns = colnames
    raw_data[label] = df

# ------------------------------
# 2) Window-based feature extraction
# ------------------------------
window_size = 100  # 1 second at 100 Hz
step_size = 50     # 0.5 second overlap

features = []
labels = []

for label, df in raw_data.items():
    data = df[channels].values.astype(float)
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data[start:start+window_size]
        feats = []
        for i in range(window.shape[1]):
            col = window[:, i]
            feats.extend([
                np.mean(col),
                np.std(col),
                np.min(col),
                np.max(col),
                np.mean(col**2)
            ])
        features.append(feats)
        labels.append(label)

X = np.array(features)
y = np.array(labels)

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
class_names = le.classes_

# ------------------------------
# 3) Split data
# ------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y_enc, test_size=0.4, random_state=42, stratify=y_enc)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# ------------------------------
# 4) Train MLP classifier
# ------------------------------
mlp = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", max_iter=200, random_state=42))
])
mlp.fit(X_train, y_train)

# ------------------------------
# 5) Evaluate metrics
# ------------------------------
y_pred = mlp.predict(X_test)
proba = mlp.predict_proba(X_test)

precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print("Evaluation Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")

# ------------------------------
# 6) ROC-AUC visualization
# ------------------------------
# Binarize labels for ROC
n_classes = len(class_names)
y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

plt.figure(figsize=(8,6))
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve")
plt.legend()
plt.grid(True)
plt.savefig("roc_auc_mlp.png", dpi=160)
plt.show()

# ------------------------------
# 7) Log Loss curve visualization
# ------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

mlp_curve = MLPClassifier(hidden_layer_sizes=(64,32), activation="relu", solver="adam", max_iter=1, warm_start=True, random_state=42)
train_ll, val_ll = [], []
EPOCHS = 100
for epoch in range(EPOCHS):
    if epoch == 0:
        mlp_curve.partial_fit(X_train_s, y_train, classes=np.arange(n_classes))
    else:
        mlp_curve.partial_fit(X_train_s, y_train)
    train_ll.append(log_loss(y_train, mlp_curve.predict_proba(X_train_s)))
    val_ll.append(log_loss(y_val, mlp_curve.predict_proba(X_val_s)))

plt.figure(figsize=(8,6))
plt.plot(range(EPOCHS), train_ll, label="Train Log Loss")
plt.plot(range(EPOCHS), val_ll, label="Validation Log Loss")
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Log Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("log_loss_mlp.png", dpi=160)
plt.show()

# ------------------------------
# 8) Visualize MLP architecture (text-based summary)
# ------------------------------
mlp_architecture = mlp.named_steps["clf"]
print("\nMLP Architecture:")
print(mlp_architecture)

# Save architecture visualization as text file
with open("architecture_mlp.txt", "w") as f:
    f.write(str(mlp_architecture))
