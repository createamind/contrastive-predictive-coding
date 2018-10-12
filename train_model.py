'''
This module describes the contrastive predictive coding model from DeepMind:

Oord, Aaron van den, Yazhe Li, and Oriol Vinyals.
"Representation Learning with Contrastive Predictive Coding."
arXiv preprint arXiv:1807.03748 (2018).
'''
from data_utils import SortedNumberGenerator
from os.path import join, basename, dirname, exists
import datetime
import argparse
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import keras
from keras import backend as K


def network_encoder(x, code_size):

    ''' Define the network mapping images to embeddings '''

    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding')(x)

    return x


def network_autoregressive(args, x):

    ''' Define the network that integrates information along the sequence '''

    # x = keras.layers.GRU(units=256, return_sequences=True)(x)
    # x = keras.layers.BatchNormalization()(x)

    if args.doctor:
        x = keras.layers.LSTM(units=256, return_states=True, return_sequences=False, name='ar_context')(x)[1]
    else:
        x = keras.layers.GRU(units=256, return_sequences=False, name='ar_context')(x)

    return x


def network_prediction(context, code_size, predict_terms):

    ''' Define the network mapping context to multiple embeddings '''

    outputs = []
    for i in range(predict_terms):
        outputs.append(keras.layers.Dense(units=code_size, activation="linear", name='z_t_{i}'.format(i=i))(context))

    if len(outputs) == 1:
        output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
    else:
        output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)

    return output


class CPCLayer(keras.layers.Layer):

    ''' Computes dot product between true and predicted embedding vectors '''

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):

        # Compute dot product among vectors
        preds, y_encoded = inputs

        dot_product = K.mean(y_encoded * preds, axis=-1)
        dot_product = K.mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension

        # Keras loss functions take probabilities
        dot_product_probs = K.sigmoid(dot_product)

        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


class MSELayer(keras.layers.Layer):

    ''' Computes dot product between true and predicted embedding vectors '''

    def __init__(self, **kwargs):
        super(MSELayer, self).__init__(**kwargs)

    def call(self, inputs):

        # Compute dot product among vectors
        preds, z_encoded = inputs
        ans = K.mean((z_encoded - preds) * (z_encoded - preds), axis=-1)
        ans = K.mean(ans, axis=-1, keepdims=True)  # average along the temporal dimension

        return ans

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


def network_cpc(args, image_shape, terms, predict_terms, code_size, learning_rate):

    ''' Define the CPC network combining encoder and autoregressive model '''

    # Set learning phase (https://stackoverflow.com/questions/42969779/keras-error-you-must-feed-a-value-for-placeholder-tensor-bidirectional-1-keras)
    K.set_learning_phase(1)

    # Define encoder model
    encoder_input = keras.layers.Input(image_shape)
    encoder_output = network_encoder(encoder_input, code_size)
    encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    encoder_model.summary()

    # Define rest of model
    x_input = keras.layers.Input((terms, image_shape[0], image_shape[1], image_shape[2]))
    x_encoded = keras.layers.TimeDistributed(encoder_model)(x_input)
    context = network_autoregressive(args, x_encoded)
    preds = network_prediction(context, code_size, predict_terms)

    y_input = keras.layers.Input((predict_terms, image_shape[0], image_shape[1], image_shape[2]))
    y_encoded = keras.layers.TimeDistributed(encoder_model)(y_input)

    # Loss
    dot_product_probs = CPCLayer()([preds, y_encoded])

    z_input = keras.layers.Input((predict_terms, image_shape[0], image_shape[1], image_shape[2]))
    z_encoded = keras.layers.TimeDistributed(encoder_model)(z_input)

    mse = MSELayer()([preds, z_encoded])

    # Model
    cpc_model = keras.models.Model(inputs=[x_input, y_input, z_input], outputs=[dot_product_probs, mse])
    # Compile model
    cpc_model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss={
            'cpc_layer_1': 'binary_crossentropy',
            'mse_layer_1': 'mean_absolute_error'
        },
        loss_weights={
            'cpc_layer_1': 1.,
            'mse_layer_1': args.mse_weight
        },
        metrics={
            'cpc_layer_1': 'binary_accuracy',
            'mse_layer_1': 'mae'
        }
    )
    cpc_model.summary()

    return cpc_model


def train_model(args, epochs, batch_size, output_dir, code_size, lr=1e-4, terms=4, predict_terms=4, image_size=28, color=False):

    # Prepare data
    train_data = SortedNumberGenerator(batch_size=batch_size, subset='train', terms=terms,
                                       positive_samples=batch_size // 2, predict_terms=predict_terms,
                                       image_size=image_size, color=color, rescale=True)

    validation_data = SortedNumberGenerator(batch_size=batch_size, subset='valid', terms=terms,
                                            positive_samples=batch_size // 2, predict_terms=predict_terms,
                                            image_size=image_size, color=color, rescale=True)

    # Prepares the model
    model = network_cpc(args, image_shape=(image_size, image_size, 3), terms=terms, predict_terms=predict_terms,
                        code_size=code_size, learning_rate=lr)

    # Callbacks
    callbacks = [
        # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4),
        keras.callbacks.TensorBoard(log_dir='./logs/train_' + args.name + '_' +datetime.datetime.now().strftime('%d_%H-%M-%S ') , histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    ]

    # Trains the model
    model.fit_generator(
        generator=train_data,
        steps_per_epoch=len(train_data),
        validation_data=validation_data,
        validation_steps=len(validation_data),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )

    # Saves the model
    # Remember to add custom_objects={'CPCLayer': CPCLayer} to load_model when loading from disk
    model.save(join(output_dir, args.name + '.h5'))

    # Saves the encoder alone
    encoder = model.layers[1].layer
    encoder.save(join(output_dir, 'encoder_' + args.name + '.h5'))


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description='CPC')
    # argparser.add_argument(
    #     '--host',
    #     metavar='H',
    #     default='localhost',
    #     help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '--name',
        default='cpc',
        help='name')
    argparser.add_argument(
        '-e', '--epochs',
        default=10,
        type=int,
        help='epochs')
    argparser.add_argument(
        '--lr',
        default=1e-3,
        type=float,
        help='Learning rate')
    # argparser.add_argument(
    #     '-i', '--image-size',
    #     default=160,
    #     type=int,
    #     help='Size of images (default: 320).')
    # argparser.add_argument(
    #     '-b', '--batch-size',
    #     default=32,
    #     type=int,
    #     help='Size of batches.')
    # argparser.add_argument(
    #     '-t', '--train-epoch',
    #     default=100,
    #     type=int,
    #     help='Times of train.')
    # argparser.add_argument(
    #     '--vaealpha',
    #     default=1,
    #     type=int,
    #     help='Times of train.')
    argparser.add_argument(
        '--mse-weight',
        default=0.01,
        type=float,
        help='Weight of MSE.')
    # argparser.add_argument(
    #     '--name',
    #     default=0.01,
    #     type=float,
    #     help='Weight of MSE.')
    argparser.add_argument('--doctor', action='store_true', default=False, help='Doctor')
        
    args = argparser.parse_args()

    if args.doctor:
        predict_terms = 1
    else:
        predict_terms = 1
    train_model(
        args, 
        epochs=args.epochs,
        batch_size=32,
        output_dir='models/64x64',
        code_size=128,
        lr=args.lr,
        terms=4,
        predict_terms=predict_terms,
        image_size=64,
        color=True
    )

