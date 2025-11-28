import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import argparse

IMAGE_DIR = "images"
IMAGE_SIZE = 128
REGIONS = ['Piedmont', 'Southeastern Plains', 'Blue Ridge', 'Ridge and Valley',
           'Southwestern Appalachians', 'Southern Coastal Plain']
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42

def parse_folder(image_dir=IMAGE_DIR, allowed=REGIONS):
    image_paths, labels = [], []
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
        if matched:
            image_paths.append(os.path.join(image_dir, fname))
            labels.append(matched)
    return image_paths, labels

def load_images(paths, image_size=IMAGE_SIZE):
    X = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            X.append(np.zeros((image_size, image_size, 3), dtype=np.float32))
            continue
        img = cv2.resize(img, (image_size, image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(img.astype(np.float32) / 255.0)
    return np.stack(X, axis=0)

def build_wide_bn_dropout(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=6, channels=[64,128,256], dropout=0.4):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(channels[0], 3, padding='same', use_bias=False),
        layers.BatchNormalization(), layers.ReLU(), layers.MaxPool2D(2),

        layers.Conv2D(channels[1], 3, padding='same', use_bias=False),
        layers.BatchNormalization(), layers.ReLU(), layers.MaxPool2D(2),

        layers.Conv2D(channels[2], 3, padding='same', use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def plot_classification_metrics(y_true, y_pred, class_names):
    # Compute precision, recall, f1 per class
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    df = pd.DataFrame({
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    df.set_index("Class", inplace=True)
    df.plot(kind='bar', figsize=(13, 8))
    plt.title("Classification Metrics per Class - CNN Classifier")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()

def main(args):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    image_paths, raw_labels = parse_folder(args.image_dir)
    print(f"Found {len(image_paths)} images.")
    if not image_paths:
        raise SystemExit("No images found.")

    X = load_images(image_paths, args.image_size)
    le = LabelEncoder()
    y = le.fit_transform(raw_labels)
    num_classes = len(le.classes_)
    y_cat = to_categorical(y, num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=args.test_size, random_state=SEED, stratify=y
    )

    model = build_wide_bn_dropout(
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=num_classes, channels=[64,128,256], dropout=args.dropout
    )
    model.compile(optimizer=optimizers.Adam(learning_rate=args.lr),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train, y_train, validation_data=(X_test, y_test),
        epochs=args.epochs, batch_size=args.batch_size
    )

    preds = model.predict(X_test)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # ---- Classification Report ----
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {overall_acc:.3f}")

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix - CNN Classifier")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # ---- Classification Metrics Bar Plot ----
    plot_classification_metrics(y_true, y_pred, le.classes_)

    # ---- Training Curves ----
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label="Train Acc")
    plt.plot(history.history['val_accuracy'], label="Val Acc")
    plt.title("Accuracy Over Training")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Val Loss")
    plt.title("Loss Over Training")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', default=IMAGE_DIR)
    parser.add_argument('--image-size', type=int, default=IMAGE_SIZE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--dropout', type=float, default=0.4)
    args = parser.parse_args()
    main(args)
