import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

# -------------------------------
# Parameters
# -------------------------------
IMAGE_DIR = "images"
REGIONS = ['Piedmont', 'Southeastern Plains', 'Blue Ridge', 'Ridge and Valley',
           'Southwestern Appalachians', 'Southern Coastal Plain']

# -------------------------------
# 1) Load image labels
# -------------------------------
image_paths = []
labels = []

for fname in os.listdir(IMAGE_DIR):
    if fname.endswith(".png"):
        region_label = "_".join(fname.split("_")[1:]).replace(".png","").replace("_"," ")
        if region_label not in REGIONS:
            continue
        image_paths.append(os.path.join(IMAGE_DIR, fname))
        labels.append(region_label)

print(f"Found {len(image_paths)} images.")

# -------------------------------
# 2) Encode labels
# -------------------------------
le = LabelEncoder()
y = le.fit_transform(labels)

# -------------------------------
# 3) Train/test split
# -------------------------------
X_dummy = np.zeros((len(y), 1))  # dummy features
X_train, X_test, y_train, y_test = train_test_split(
    X_dummy, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 4) Random predictions
# -------------------------------
np.random.seed(42)
y_pred = np.random.choice(len(REGIONS), size=len(y_test))  # randomly pick class indices

# -------------------------------
# 5) Confusion matrix visualization
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(14, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    cbar=True,
    annot_kws={"size": 12}
)
plt.xlabel('Predicted', fontsize=14, labelpad=10)
plt.ylabel('Actual', fontsize=14, labelpad=10)
plt.title('Confusion Matrix - Random Classifier', fontsize=16, pad=20)
plt.xticks(rotation=35, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # leaves room for the title
plt.show()

# -------------------------------
# 6) Overall accuracy
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy:.2f}")

# -------------------------------
# 7) Precision, Recall, F1 visualization
# -------------------------------
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report = df_report.iloc[:-3]  # remove accuracy/macro avg/weighted avg

plt.figure(figsize=(14, 8))
ax = df_report[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(14, 8), width=0.8)
plt.title("Classification Metrics per Class - Random Classifier", fontsize=16, pad=20)
plt.ylabel("Score", fontsize=14)
plt.ylim(0, 1)
plt.xticks(rotation=35, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



# -------------------------------
# Per-class metrics and averages
# -------------------------------
report = classification_report(
    y_test,
    y_pred,
    target_names=le.classes_,
    output_dict=True
)

# Extract per-class metrics
for region in le.classes_:
    precision = report[region]['precision']
    recall = report[region]['recall']
    f1 = report[region]['f1-score']
    support = report[region]['support']
    print(f"{region}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}, Support={support}")

# Macro averages
macro_precision = report['macro avg']['precision']
macro_recall = report['macro avg']['recall']
macro_f1 = report['macro avg']['f1-score']

print(f"\nMacro averages: Precision={macro_precision:.2f}, Recall={macro_recall:.2f}, F1={macro_f1:.2f}")

# Weighted averages
weighted_precision = report['weighted avg']['precision']
weighted_recall = report['weighted avg']['recall']
weighted_f1 = report['weighted avg']['f1-score']

print(f"Weighted averages: Precision={weighted_precision:.2f}, Recall={weighted_recall:.2f}, F1={weighted_f1:.2f}")