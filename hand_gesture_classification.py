import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv1D, MaxPool1D, LSTM, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

import mlflow
import mlflow.tensorflow

mlflow.set_experiment("Directional_Hand_Gesture_Detection")

if mlflow.active_run():
    mlflow.end_run()

def get_frames(df, frame_size, hop_size):

    N_FEATURES = 3

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['ACC_X'].values[i: i + frame_size]
        y = df['ACC_Y'].values[i: i + frame_size]
        z = df['ACC_Z'].values[i: i + frame_size]
        
        # Retrieve the most often used label in this segment
        label = df['GESTURE'][i: i + frame_size].mode()[0]
        frames.append([x, y, z])
        labels.append(label)

#     print(frames[0])
    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)
#     print(frames[0])

    return frames, labels

def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    print(f"Data loaded from {data_path}. Shape: {df.shape}")
    print("First few rows of the dataset:")
    print(df.head())

    df.columns = df.columns.str.strip()
    df = df.drop(['INDEX', 'TIMESTAMP'], axis = 1).copy()

    encoder = LabelEncoder()
    df['GESTURE'] = encoder.fit_transform(df['GESTURE'])

    X = df[['ACC_X', 'ACC_Y', 'ACC_Z']]
    y = df['GESTURE']
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaled_X = pd.DataFrame(data = X, columns = ['ACC_X', 'ACC_Y', 'ACC_Z'])
    scaled_X['GESTURE'] = y.values

    print(scaled_X.head())
    mean = scaler.mean_
    std = scaler.scale_

    print(mean)
    print(std)

    Fs = 25
    frame_size = 50
    hop_size = 25

    X, y = get_frames(scaled_X, frame_size, hop_size)

    print(f"Frames shape: {X.shape}, Labels shape: {y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # X_train = np.expand_dims(X_train, axis=-1)
    # X_test = np.expand_dims(X_test, axis=-1)
    
    # One-hot encode targets for categorical cross-entropy
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    
    return X_train, X_test, y_train_cat, y_test_cat, y_test

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

def train_and_log_pipeline():
    data_path = "./datasets/acc_data_updated.csv" 
    
    X_train, X_test, y_train, y_test_cat, y_test_raw = load_and_preprocess_data(data_path)
    
    epochs = 30
    batch_size = 32
    learning_rate = 0.001
    
    mlflow.tensorflow.autolog(log_models=True)
    
    with mlflow.start_run(run_name="TensorFlow_2D_CNN_Training") as run:
        
        mlflow.log_param("framework", "TensorFlow Keras")
        mlflow.log_param("data_samples_count", X_train.shape[0])
        
        model = build_model(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])
        model.compile(optimizer=Adam(learning_rate=learning_rate), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        
        model.fit(X_train, y_train, 
                  epochs=epochs, 
                  batch_size=batch_size, 
                  validation_data=(X_test, y_test_cat),
                  verbose=1)
        
        print("Evaluating model performance on test dataset...")
        y_pred_cat = model.predict(X_test)
        y_pred = np.argmax(y_pred_cat, axis=1)
        
        test_accuracy = np.mean(y_pred == y_test_raw)
        mlflow.log_metric("final_test_accuracy", test_accuracy)
        
        report_path = "./mlflow/classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(y_test_raw, y_pred))
        mlflow.log_artifact(report_path)
        # os.remove(report_path) 
        

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.experimental_new_converter = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS
            # tf.lite.OpsSet.SELECT_TF_OPS
        ]
        # converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        
        os.makedirs("./models", exist_ok=True)
        tflite_filename = "./models/Gesture_Classifier_model2.tflite"
        with open(tflite_filename, "wb") as f:
            f.write(tflite_model)
            
        mlflow.log_artifact(tflite_filename, artifact_path="edge_models")

if __name__ == "__main__":
    train_and_log_pipeline()