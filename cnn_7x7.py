# cnn_baseline_7x7.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import argparse

# -------------------------------
# Params
# -------------------------------
IMAGE_DIR = "images"
IMAGE_SIZE = 128
REGIONS = ['Piedmont', 'Southeastern Plains', 'Blue Ridge', 'Ridge and Valley',
           'Southwestern Appalachians', 'Southern Coastal Plain']
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42

# -------------------------------
# Utilities: load images & labels
# -------------------------------
def parse_folder(image_dir=IMAGE_DIR, allowed=REGIONS):
    image_paths = []
    labels = []
    for fname in os.listdir(image_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        parts = fname.rsplit('.', 1)[0].split('_')
        region_label = ' '.join(parts[1:]).strip()
        matched = None
        for r in allowed:
            if r.lower() in region_label.lower():
                matched = r
                break
        if matched is None:
            continue
        image_paths.append(os.path.join(image_dir, fname))
        labels.append(matched)
    return image_paths, labels

def load_images(paths, image_size=IMAGE_SIZE):
    X = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print("Warning: failed to read", p)
            X.append(np.zeros((image_size, image_size, 3), dtype=np.float32))
            continue
        img = cv2.resize(img, (image_size, image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        X.append(img)
    return np.stack(X, axis=0)

# -------------------------------
# Model: Baseline 7x7 conv blocks
# -------------------------------
def build_baseline_7x7(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=6, channels=[32,64,128], kernel=7):
    model = models.Sequential()
    # conv blocks using stride=2 (to reduce size as requested) and uniform kernel size
    model.add(layers.Input(shape=input_shape))
    for ch in channels:
        model.add(layers.Conv2D(ch, kernel_size=kernel, strides=2, padding='same'))
        model.add(layers.ReLU())
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# -------------------------------
# Main
# -------------------------------
def main(args):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    image_paths, raw_labels = parse_folder(args.image_dir)
    print(f"Found {len(image_paths)} images.")
    if len(image_paths) == 0:
        raise SystemExit("No images found - check IMAGE_DIR and filename format.")

    X = load_images(image_paths, image_size=args.image_size)
    le = LabelEncoder()
    y = le.fit_transform(raw_labels)
    num_classes = len(le.classes_)
    y_cat = to_categorical(y, num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=args.test_size, random_state=SEED, stratify=y
    )

    model = build_baseline_7x7(input_shape=(args.image_size, args.image_size, 3),
                               num_classes=num_classes,
                               channels=[32,64,128],
                               kernel=7)
    model.compile(optimizer=optimizers.Adam(learning_rate=args.lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=args.epochs,
                        batch_size=args.batch_size)

    # Evaluate and print report
    preds = model.predict(X_test, batch_size=args.batch_size)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix - Baseline 7x7')
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', default=IMAGE_DIR)
    parser.add_argument('--image-size', type=int, default=IMAGE_SIZE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()
    main(args)
