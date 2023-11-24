from tensorflow import keras

model = keras.models.load_model('keras_model.h5')

model.summary()

print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)


config = model.get_config()
print(config)

print("Loss:", model.losses)
