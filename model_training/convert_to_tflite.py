import tensorflow as tf

# 1. Load the acquired H5 model
model = tf.keras.models.load_model('best_TamangASL_120k.keras')

# 2. Convert it to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3. Save the new TFLite file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Successfully converted! You can now use this in your Flet app.")