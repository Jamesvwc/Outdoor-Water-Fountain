import tensorflow as tf
# import matplotlib.pyplot as plt
# print("Tensorflow version is", tf.__version__);

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy')>0.90):
            print("\nReach 95% accuracy, cancelling the remainder of training")
            self.model.stop_training = True

callbacks = myCallback()

mnist = tf.keras.datasets.fashion_mnist; #Dataset of clothing from Keras

(training_images, training_labels), (test_images, test_labels) = mnist.load_data();

# plt.imshow(training_images[1])
# print(training_labels[1])
# print(training_images[1])
assert training_images.shape == (60000, 28, 28)
assert training_labels.shape == (60000,)
assert test_images.shape == (10000, 28, 28)
assert test_labels.shape == (10000,)

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# print(training_labels[1])
# print(training_images[1])

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(training_images, training_labels, epochs=15, callbacks=[callbacks])