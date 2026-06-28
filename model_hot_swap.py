import threading

import numpy as np
import tensorflow as tf

class LiveModelContainer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._lock = threading.Lock()
        self.load_model()

    def load_model(self):
        with self._lock:
            print(f"Loading active model binary from: {self.model_path}")
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

    def predict(self, input_data):
        with self._lock:
            input_tensor = np.array(input_data, dtype=np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            prediction_probabilities = self.interpreter.get_tensor(self.output_details[0]['index'])
            predicted_gesture_id = int(np.argmax(prediction_probabilities))
            confidence_score = float(prediction_probabilities[predicted_gesture_id])
            return predicted_gesture_id, confidence_score