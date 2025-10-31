import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# Parameters
# -------------------------------
IMAGE_DIR = "images"
REGIONS = ['Piedmont', 'Southeastern Plains', 'Blue Ridge', 'Ridge and Valley',
           'Southwestern Appalachians', 'Southern Coastal Plain']

# -------------------------------
# 1) Load images and labels
# -------------------------------
image_paths = []
labels = []

for fname in os.listdir(IMAGE_DIR):
    if fname.endswith(".png"):
        path = os.path.join(IMAGE_DIR, fname)
        # Correct label parsing: join everything after the first underscore
        region_label = "_".join(fname.split("_")[1:]).replace(".png","").replace("_"," ")
        if region_label not in REGIONS:
            continue
        image_paths.append(path)
        labels.append(region_label)

print(f"Found {len(image_paths)} images.")

# -------------------------------
# 2) Extract features (color histograms)
# -------------------------------
def extract_color_histogram(image_path, bins=(8,8,8)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Failed to read {image_path}")
        return np.zeros(np.prod(bins), dtype=float)  # return empty feature
    image = cv2.resize(image, (128,128))  # resize to uniform size
    hist = cv2.calcHist([image], [0,1,2], None, bins, [0,256,0,256,0,256])  # <-- 6 values, not 8
    hist = cv2.normalize(hist, hist).flatten()
    return hist


features = np.array([extract_color_histogram(p) for p in image_paths])
print(f"Feature array shape: {features.shape}")

# -------------------------------
# 3) Encode labels
# -------------------------------
le = LabelEncoder()
y = le.fit_transform(labels)

# -------------------------------
# 4) Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 5) Train SVM
# -------------------------------
svm = SVC(kernel='linear', C=1.0, probability=True)
svm.fit(X_train, y_train)

# -------------------------------
# 6) Evaluate
# -------------------------------
y_pred = svm.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
