import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

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

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Classifier')
plt.show()

# -------------------------------
# 6) Precision, Recall, F1 visualization
# -------------------------------
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report = df_report.iloc[:-3]  # remove accuracy/macro avg/weighted avg

df_report[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(10,6))
plt.title("Classification Metrics per Class - Random Classifier")
plt.ylabel("Score")
plt.ylim(0,1)
plt.xticks(rotation=45)
plt.show()
