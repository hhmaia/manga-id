import os, random
import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError

from src import utils

os.umask(0)
train_dataset_dir = os.path.join('/datasets/manga/')
val_dataset_dir = os.path.join('/datasets/manga-val/')
ckp_filepath = os.path.join('/opt/project/ckp/02')

def get_model():
    model_01 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(450, 300, 1)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model_02 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='selu', input_shape=(450, 300, 1)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='selu', kernel_regularizer='l2'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='selu', kernel_regularizer='l2'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='selu', kernel_regularizer='l2'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='selu', kernel_regularizer='l2'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='selu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    try:
        model_02.load_weights(ckp_filepath)
        print('checkpoint loaded!')
    except NotFoundError:
        print('no checkpoint found on ' + ckp_filepath)
    except ValueError:
        print('no checkpoint found on ' + ckp_filepath)

    model_02.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_02.summary()
    return model_02

class EarlyStopCB(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.95:
            model.stop_training = True
early_stop_cb = EarlyStopCB()

ckp_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=ckp_filepath,
    save_weights_only=True,
    monitor='accuracy',
    mode='max',
    save_best_only=True)

model = get_model()

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255.,
    height_shift_range=0.1,
    width_shift_range=0.1,
    zoom_range=0.1,
    # brightness_range=0.2,
    rotation_range=0.1,
    horizontal_flip=True,
    # validation_split=0.2
)

train_gen = data_generator.flow_from_directory(
    train_dataset_dir,
    target_size=(450, 300),
    batch_size=50,
    seed=1,
    color_mode='grayscale',
    class_mode='sparse',
    # subset='training'
)

val_gen = data_generator.flow_from_directory(
    val_dataset_dir,
    target_size=(450, 300),
    batch_size=50,
    seed=1,
    color_mode='grayscale',
    class_mode='sparse',
    # subset='validation'
)

if True:
    hist = model.fit(
        train_gen,
        steps_per_epoch=61,
        epochs=50,
        validation_data=val_gen,
        validation_steps=12,
        callbacks=[ckp_cb])#, early_stop_cb])

    utils.plot_hist(hist, 'accuracy', '/opt/project/data/', 'test3')
    utils.plot_hist(hist, 'loss', '/opt/project/data/', 'test3')

utils.create_features_visuals(
    model,
    #["/datasets/manga-test/Berserk_360_2.png"],
    ["/datasets/manga-test/biomega1.jpg"],
    (450, 300),
    '/opt/project/data/vis/')

print('exiting...')
