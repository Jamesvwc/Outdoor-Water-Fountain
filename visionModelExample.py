import tensorflow as tf
import matplotlib.pyplot as plt
print("Tensorflow version is", tf.__version__);

mnist = tf.keras.datasets.fashion_mnist; #Dataset of clothing from Keras

(training_images, training_labels), (test_images, test_labels) = mnist.load_data();

plt.imshow(training_images[1])
print(training_labels[1])
print(training_images[1])

training_images = training_images / 255.0
training_labels = training_labels / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

print(training_labels[1])
print(training_images[1])