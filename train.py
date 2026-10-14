"""サイコロのピンゾロ判定AI: モデルの学習部分

データの読み込み・モデル定義は他メンバーの担当部分を想定し、
このファイルでは「学習 → 評価 → 学習曲線の表示」を行う関数を提供する。

使い方 (他メンバーのコードと結合する場合):
    from train import train_and_evaluate
    history, loss, accuracy = train_and_evaluate(model, x_train, y_train, x_test, y_test)

単体で動作確認する場合:
    python train.py   # ダミーデータと簡易モデルで学習の流れだけ確認できる
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


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


def _build_dummy_model(input_shape):
    """動作確認用の簡易CNN (本番では担当メンバーのモデルに差し替える)。"""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),  # 1: ピンゾロ, 0: それ以外
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    # ダミーデータ (64x64 のグレースケール画像, ラベル 0/1)
    rng = np.random.default_rng(0)
    x_train = rng.random((200, 64, 64, 1), dtype=np.float32)
    y_train = rng.integers(0, 2, 200)
    x_test = rng.random((50, 64, 64, 1), dtype=np.float32)
    y_test = rng.integers(0, 2, 50)

    model = _build_dummy_model(x_train.shape[1:])
    train_and_evaluate(model, x_train, y_train, x_test, y_test, epochs=5)
