import os
import time
from fastapi import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from sqlalchemy import text
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import mlflow
import mlflow.tensorflow
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv1D, MaxPool1D, LSTM, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from database import engine

CSV_BASE_DATASET_PATH = "./datasets/acc_data_updated.csv" 
DATA_LAKE_DIR = "data_lake_storage"
FRAME_SIZE = 50
HOP_SIZE = 25
N_FEATURES = 3

def get_frames_scheduler(df, frame_size, hop_size):
    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['ACC_X'].values[i: i + frame_size]
        y = df['ACC_Y'].values[i: i + frame_size]
        z = df['ACC_Z'].values[i: i + frame_size]
        
        label = df['GESTURE'][i: i + frame_size].mode()[0]
        frames.append([x, y, z])
        labels.append(label)

    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)
    return frames, labels

def load_and_process_old_data(data_path):
    """Your exact cleaning and scaling pipeline to generate baseline 3D boxes"""
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    df = df.drop(['INDEX', 'TIMESTAMP'], axis=1).copy()

    # Fit and transform categorical and scalar boundaries
    encoder = LabelEncoder()
    df['GESTURE'] = encoder.fit_transform(df['GESTURE'])

    X = df[['ACC_X', 'ACC_Y', 'ACC_Z']]
    y = df['GESTURE']
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaled_X = pd.DataFrame(data=X, columns=['ACC_X', 'ACC_Y', 'ACC_Z'])
    scaled_X['GESTURE'] = y.values

    # Run sliding window segmentation
    X_old_3d, y_old_3d = get_frames_scheduler(scaled_X, FRAME_SIZE, HOP_SIZE)
    return X_old_3d, y_old_3d

def build_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool1D(pool_size=2, padding='same'),
        Dropout(0.2),
        
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool1D(pool_size=2, padding='same'),
        Dropout(0.3),

        LSTM(64, return_sequences=True, unroll=True),
        Dropout(0.3),
        LSTM(64, unroll=True),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def execute_automated_retraining_pipeline():
    print(f"[{datetime.now()}] Scanning retraining pool metrics...")
    
    with engine.connect() as connection:
        query = text("SELECT * FROM high_confidence_retrain_pool WHERE is_processed = FALSE;")
        df = pd.read_sql_query(query, connection)
        
    if len(df) < 100:  # Context Threshold: Require at least 100 samples to justify a retraining run
        print(f"Retraining pool data density insufficient ({len(df)}/100). Postponing execution.")
        return

    print(f"Data threshold satisfied! Initializing pipeline run on {len(df)} production samples...")

    print("Processing historical CSV data...")
    if os.path.exists(CSV_BASE_DATASET_PATH):
        X_old, y_old = load_and_process_old_data(CSV_BASE_DATASET_PATH)
        print(f"Base 3D Matrix Shape: {X_old.shape}")
    else:
        print(f"Error: Baseline data not found at {CSV_BASE_DATASET_PATH}")
        return
    
    print("Compiling new Data...")
    new_features = []
    new_labels = []
    processed_pool_ids = []

    for _, row in df.iterrows():
        file_path = row['raw_data_key']
        if file_path and os.path.exists(file_path):
            try:
                matrix = np.load(file_path) 
                new_features.append(matrix)
                new_labels.append(row['predicted_gesture_id'])
                processed_pool_ids.append(row['pool_id'])
            except Exception as e:
                print(f"Skipping corrupted file {file_path}: {e}")

    X_new = np.stack(new_features, axis=0) # Shape: (Num_New_Logs, 50, 3)
    y_new = np.array(new_labels, dtype=np.int32)
    print(f"New data Shape: {X_new.shape}")

    print("Merging old 3D arrays and new 3D arrays together...")
    X_combined = np.concatenate([X_old, X_new], axis=0)
    y_combined = np.concatenate([y_old, y_new], axis=0)
    print(f"Merged Dataset Size: {X_combined.shape[0]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
    )

    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    mlflow.set_experiment("Automated_Production_Retraining_Pipeline")
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name=f"Auto_Retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("samples_retrained_count", len(df))
        
        model = build_model(input_shape=X_train.shape[1:], num_classes=y_train_cat.shape[1])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1, validation_data=(X_test, y_test_cat))
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
        
        tflite_path = "./models/retrained_production_model.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        requests.post('http://127.0.0.1:8000/api/v1/analytics/refresh')
            
        mlflow.log_artifact(tflite_path, artifact_path="deployed_production_models")
        # os.remove(tflite_path)
        print("New TFLite compiled binary successfully registered inside MLflow pipeline registry!")

    with engine.connect() as connection:
        trans = connection.begin()
        try:
            pool_ids = tuple(df['pool_id'].tolist())
            # Handle standard SQL formatting for single item lists or multi tuples safety
            if len(pool_ids) == 1:
                connection.execute(text(f"UPDATE high_confidence_retrain_pool SET is_processed = TRUE WHERE pool_id = {pool_ids[0]};"))
            else:
                connection.execute(text(f"UPDATE high_confidence_retrain_pool SET is_processed = TRUE WHERE pool_id IN {pool_ids};"))
            trans.commit()
            print("Production pool database pointers successfully reset.")
        except Exception as e:
            trans.rollback()
            print(f"Error resetting database pointers: {str(e)}")

def start_mlops_scheduler():
    scheduler = BackgroundScheduler()
    # Scans the database pool table every hour
    scheduler.add_job(execute_automated_retraining_pipeline, 'interval', hours=1, id='retrain_job')
    scheduler.start()
    print("Background MLOps scheduler engine is now running.")