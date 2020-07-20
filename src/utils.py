import os, random
import numpy as np
import tensorflow as tf
import tensorflow.keras.preprocessing.image as kimage
import matplotlib.pyplot as plt


def plot_hist(history, key, output_path, sufix=''):
    epochs = range(len(history.history[key]))
    train_hist_data = history.history[key]
    val_hist_data = history.history['val_' + key]
    plt.plot(epochs, train_hist_data, color='red')
    plt.plot(epochs, val_hist_data, color='blue')
    plt.title('Train and validation ' + key)
    plt.xlabel("Epochs")
    plt.ylabel(key)
    plt.legend([key, 'val_' + key])
    plt.savefig(''.join([output_path, key, '-', sufix]))
    plt.clf()


def postprocess_filter_output(filter_output):
    x = filter_output
    x -= x.mean()
    x /= x.std()
    x *= 64
    x += 128
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def create_features_visuals(model: tf.keras.models.Model,
                            samples_fpaths: list,
                            target_size,
                            output_path: str,
                            sufix: str = ''):
    # select a random image and prepare it for input on the model
    random_fname = random.choice(samples_fpaths)
    img = kimage.load_img(random_fname, color_mode='grayscale', target_size=target_size)
    img = kimage.img_to_array(img)
    img /= 255
    img = np.reshape(img, [1, target_size[0], target_size[1], -1])
    # expose all the outputs from the model on a new model
    layers_outputs = [layer.output for layer in model.layers]
    layer_names = [layer.name for layer in model.layers]
    prediction_model = tf.keras.models.Model(inputs=model.inputs, outputs=layers_outputs)
    # get the output from every layer
    prediction_outputs = prediction_model.predict(img)

    for layer_name, layer_output in zip(layer_names, prediction_outputs):
        plt.clf()
        if len(layer_output.shape) == 4:
            n_filters = layer_output.shape[-1]
            for filter_index in range(n_filters):
                filter_output = layer_output[0, :, :, filter_index]
                x = postprocess_filter_output(filter_output)
                ax = plt.subplot(8, 8, filter_index + 1)
                ax.axis('off')
                plt.imshow(x, cmap='viridis')
            plt.savefig(os.path.join(output_path, layer_name + '-' + sufix))
    plt.clf()
    # plt.subplot(1, 1, 1)
    # plt.imshow(np.reshape(prediction_outputs[-2], [1, 128]), cmap='viridis')
    # plt.savefig('/opt/project/data/vis/dense1' + '-' + sufix)
