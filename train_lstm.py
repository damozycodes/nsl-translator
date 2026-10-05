"""Train a recognition model on a frozen recording-disjoint split.

Local: python train_lstm.py
Colab: upload the project, cd into it, then run this script (no hidden drive mount).
The test partition is never loaded by training. Use evaluation.py afterwards.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from config import (PROJECT_ROOT, SPLIT_PATH, MAX_EPOCHS, BATCH_SIZE, RANDOM_SEED,
                    EARLY_STOP_PATIENCE, AUG_NOISE_SIGMA, AUG_TIMEWARP_FACTOR)
from dataset_utils import load_split, file_hash


def add_gaussian_noise(keypoints, sigma=AUG_NOISE_SIGMA, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    noisy = keypoints.copy()
    # Keep absent body parts and padded frames absent; visibility is not a coordinate.
    for start, end in [(0,132), (132,1536), (1536,1599), (1599,1662)]:
        visible = np.any(keypoints[:,start:end] != 0, axis=1)
        noise = rng.normal(0, sigma, (int(visible.sum()), end-start)).astype(keypoints.dtype)
        noisy[visible,start:end] += noise
    noisy[:,3:132:4] = keypoints[:,3:132:4]
    return noisy


def time_warp(keypoints, factor=AUG_TIMEWARP_FACTOR, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    result = np.zeros_like(keypoints)
    valid = np.flatnonzero(np.any(keypoints != 0, axis=1))
    if not len(valid):
        return result
    n = valid[-1]+1
    # Nearest-frame resampling avoids synthesizing landmarks for missing detections.
    # Warp speed inside the sign while preserving both ends of its motion.
    positions = (np.linspace(0, 1, n) ** rng.uniform(1-factor,1+factor) * (n-1)).round().astype(int)
    result[:n] = keypoints[positions]
    return result


def build_model(n_classes):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, Masking, LSTM, Dense, Dropout
    model = Sequential([Input(shape=(30,1662)), Masking(mask_value=0.0),
                        LSTM(64, return_sequences=True, dropout=0.2),
                        LSTM(128, return_sequences=True, dropout=0.2),
                        LSTM(64, dropout=0.2), Dense(64,activation='relu'),
                        Dropout(0.5), Dense(n_classes,activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train(args):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    split = json.loads(args.split.read_text())
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise ValueError('Run directory is not empty. Select a new --run-dir to preserve earlier results.')
    X, y = load_split(split, 'train')
    X_val, y_val = load_split(split, 'validation')
    rng = np.random.default_rng(RANDOM_SEED)
    if not args.no_augmentation:
        X = np.concatenate([X, np.stack([add_gaussian_noise(a,rng=rng) for a in X]),
                            np.stack([time_warp(a,rng=rng) for a in X])])
        y = np.tile(y,3)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir/'label_map.json').write_text(json.dumps(split['label_map'],indent=2))
    (args.run_dir/'split_manifest.json').write_text(args.split.read_text())
    metadata = {'epochs_requested': args.epochs, 'seed': RANDOM_SEED,
                'split_sha256': file_hash(args.split), 'tensorflow': tf.__version__,
                'training_samples': len(X), 'validation_samples': len(X_val),
                'architecture': 'MediaPipe keypoints + 3-layer LSTM recognition',
                'status': 'training'}
    (args.run_dir/'run.json').write_text(json.dumps(metadata,indent=2))
    model = build_model(len(split['label_map']))
    # Explicit datasets prevent Keras from creating unbounded private thread pools.
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    training = tf.data.Dataset.from_tensor_slices((X,y)).shuffle(len(y), seed=RANDOM_SEED).batch(args.batch_size).with_options(options)
    validation = tf.data.Dataset.from_tensor_slices((X_val,y_val)).batch(args.batch_size).with_options(options)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE,
                                                   restore_best_weights=True),
                 tf.keras.callbacks.ModelCheckpoint(str(args.run_dir/'best.keras'),monitor='val_loss',save_best_only=True),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=5,factor=.5),
                 tf.keras.callbacks.CSVLogger(str(args.run_dir/'epochs.csv'))]
    history = model.fit(training,validation_data=validation,epochs=args.epochs,callbacks=callbacks,verbose=2)
    model.save(args.run_dir/'final.keras')
    (args.run_dir/'training_history.json').write_text(json.dumps(history.history,indent=2))
    metadata['status'] = 'complete'
    metadata['epochs_completed'] = len(history.history['loss'])
    (args.run_dir/'run.json').write_text(json.dumps(metadata,indent=2))
    print(f'Saved recognition model to {args.run_dir}. Test data has not been evaluated.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', type=Path, default=SPLIT_PATH)
    parser.add_argument('--run-dir', type=Path, default=PROJECT_ROOT/'models/baseline')
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--no-augmentation', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch size must be positive')
    train(args)


if __name__ == '__main__':
    main()
