# cnn_deep_3x3_pool_metrics.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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

def build_deep_3x3_pool(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=6, channels=[32,64,128]):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv2D(channels[0], 3, padding='same')); model.add(layers.ReLU())
    model.add(layers.Conv2D(channels[0], 3, padding='same')); model.add(layers.ReLU())
    model.add(layers.MaxPool2D(2))

    model.add(layers.Conv2D(channels[1], 3, padding='same')); model.add(layers.ReLU())
    model.add(layers.Conv2D(channels[1], 3, padding='same')); model.add(layers.ReLU())
    model.add(layers.MaxPool2D(2))

    model.add(layers.Conv2D(channels[2], 3, padding='same')); model.add(layers.ReLU())
    model.add(layers.Conv2D(channels[2], 3, padding='same')); model.add(layers.ReLU())
    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def plot_training_curves(history):
    # Accuracy
    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Loss
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_per_class_f1(target_names, f1_scores):
    plt.figure(figsize=(10,5))
    sns.barplot(x=target_names, y=f1_scores)
    plt.title("Per-Class F1 Scores")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("F1 Score")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()

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
    plt.title("Classification Metrics per Class - Deep 3x3 Pool CNN")
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

    model = build_deep_3x3_pool(
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=num_classes
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    plot_training_curves(history)

    preds = model.predict(X_test, batch_size=args.batch_size)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    # Macro/weighted metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    print("\n==== Extra Metrics ====")
    print(f"Macro Precision:  {precision_macro:.4f}")
    print(f"Macro Recall:     {recall_macro:.4f}")
    print(f"Macro F1:         {f1_macro:.4f}")
    print(f"Weighted Precision:  {precision_weighted:.4f}")
    print(f"Weighted Recall:     {recall_weighted:.4f}")
    print(f"Weighted F1:         {f1_weighted:.4f}")

    # Per-class F1 bar chart
    _, _, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    plot_per_class_f1(le.classes_, f1_per_class)

    # Classification Metrics Bar Plot (Precision, Recall, F1 together)
    plot_classification_metrics(y_true, y_pred, le.classes_)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Deep 3x3 Pool')
    plt.tight_layout()
    plt.show()

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