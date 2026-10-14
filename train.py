"""サイコロのピンゾロ判定AI: モデルの学習部分

DiceDataset (1〜6 の目ごとにフォルダ分けされたサイコロ1個の画像) で
目を6クラス分類するモデルを学習し、全部のサイコロが1ならピンゾロと判定する。

使い方 (他メンバーのコードと結合する場合):
    from train import train_and_evaluate
    history, loss, accuracy = train_and_evaluate(model, x_train, y_train, x_test, y_test)

単体で学習する場合:
    python train.py path/to/DiceDataset --epochs 10
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


def plot_history(loss, accuracy, val_loss=None, val_accuracy=None, save_path="history.png"):
    """エポックごとの loss と accuracy を左右2つのグラフに描画する。"""
    epochs = range(1, len(loss) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(epochs, loss, marker="o", label="train")
    if val_loss is not None:
        ax_loss.plot(epochs, val_loss, marker="o", label="validation")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()

    ax_acc.plot(epochs, accuracy, marker="o", label="train")
    if val_accuracy is not None:
        ax_acc.plot(epochs, val_accuracy, marker="o", label="validation")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"学習曲線を {save_path} に保存しました")
    plt.show()
    plt.close(fig)


def train_and_evaluate(model, x_train, y_train, x_test, y_test,
                       epochs=5, batch_size=32, validation_split=0.2):
    """モデルを学習し、テストデータで評価して学習曲線を表示する。

    model は compile 済みで、metrics に "accuracy" を含んでいること。
    """
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,  # 過学習していないか確認するため一部を検証用に分ける
    )

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"テストデータ: loss = {loss:.4f}, accuracy = {accuracy:.4f}")

    plot_history(
        history.history["loss"],
        history.history["accuracy"],
        history.history.get("val_loss"),
        history.history.get("val_accuracy"),
    )
    return history, loss, accuracy


def load_dataset(dataset_dir, test_ratio=0.2, seed=0):
    """DiceDataset/{1..6}/*.bmp を読み込み、学習用とテスト用に分ける。

    画像は 0〜1 に正規化したグレースケール (128, 128, 1)。
    ラベルはサイコロの目 - 1 (0〜5)。
    """
    images, labels = [], []
    for face in range(1, 7):
        for path in sorted(glob.glob(os.path.join(dataset_dir, str(face), "*.bmp"))):
            img = Image.open(path).convert("L")
            images.append(np.asarray(img, dtype=np.float32) / 255.0)
            labels.append(face - 1)
    if not images:
        raise FileNotFoundError(f"{dataset_dir} に画像が見つかりません")

    x = np.expand_dims(np.stack(images), -1)
    y = np.array(labels)

    # シャッフルしてからテスト用を切り出す (validation_split は末尾を使うため先にシャッフルが必要)
    idx = np.random.default_rng(seed).permutation(len(x))
    x, y = x[idx], y[idx]
    n_test = int(len(x) * test_ratio)
    return x[n_test:], y[n_test:], x[:n_test], y[:n_test]


def build_model(input_shape=(128, 128, 1), num_classes=6):
    """サイコロの目 (1〜6) を分類するCNN。"""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        # サイコロは向きがバラバラなので、回転・反転で学習データを水増しする (学習時のみ有効)
        tf.keras.layers.RandomRotation(0.5),
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def is_pinzoro(model, dice_images):
    """複数のサイコロ画像を受け取り、すべて1の目ならピンゾロ (True) と判定する。"""
    x = np.asarray(dice_images, dtype=np.float32)
    faces = np.argmax(model.predict(x, verbose=0), axis=1) + 1
    return bool(np.all(faces == 1)), faces.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="DiceDataset フォルダのパス")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save", default="dice_model.keras", help="学習済みモデルの保存先")
    args = parser.parse_args()

    x_train, y_train, x_test, y_test = load_dataset(args.dataset_dir)
    print(f"学習: {len(x_train)} 枚, テスト: {len(x_test)} 枚")

    model = build_model(x_train.shape[1:])
    train_and_evaluate(model, x_train, y_train, x_test, y_test, epochs=args.epochs)
    model.save(args.save)
    print(f"学習済みモデルを {args.save} に保存しました")

    # ピンゾロ判定の例: テスト画像から3個ずつ取り出して判定
    ones = x_test[y_test == 0][:3]
    others = np.concatenate([x_test[y_test == 0][3:5], x_test[y_test == 3][:1]])
    print("1,1,1 の画像 →", is_pinzoro(model, ones))
    print("1,1,4 の画像 →", is_pinzoro(model, others))
