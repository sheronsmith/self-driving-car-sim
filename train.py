"""
Train the nvidia self-driving model on simulator data.

    python train.py --data_dir data/ --epochs 10
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

from model import nvidia_model
from utils import load_log, generator


def save_loss_plot(history, path='loss.png'):
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    print(f'saved loss plot -> {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--epochs',   type=int,   default=10)
    ap.add_argument('--batch',    type=int,   default=64)
    ap.add_argument('--lr',       type=float, default=1e-4)
    ap.add_argument('--out',      default='model.keras')
    args = ap.parse_args()

    print(f'loading {args.data_dir} ...')
    paths, angles = load_log(args.data_dir)
    print(f'{len(paths)} samples (with left/right cameras)')

    # hold back 15% for validation — the network never trains on these
    X_tr, X_val, y_tr, y_val = train_test_split(
        paths, angles, test_size=0.15, random_state=42)
    print(f'train {len(X_tr)} | val {len(X_val)}')

    train_gen = generator(X_tr,  y_tr,  args.batch, training=True)
    val_gen   = generator(X_val, y_val, args.batch, training=False)

    model = nvidia_model()
    model.compile(optimizer=Adam(learning_rate=args.lr), loss='mse')
    model.summary()

    hist = model.fit(
        train_gen,
        steps_per_epoch  = len(X_tr)  // args.batch,
        validation_data  = val_gen,
        validation_steps = len(X_val) // args.batch,
        epochs = args.epochs,
        verbose = 1)

    model.save(args.out)
    print(f'model saved -> {args.out}')
    save_loss_plot(hist)


if __name__ == '__main__':
    main()
