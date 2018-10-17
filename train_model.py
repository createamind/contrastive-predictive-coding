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
from keras.models import Sequential, Model
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.optimizers import RMSprop

from tqdm import tqdm
import random
from functools import partial
import numpy as np
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import os
if not os.path.exists('models/'):
    os.makedirs('models/')

if not os.path.exists('fig'):
    os.makedirs('fig')


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def __init__(self, batch_size, predict_terms):
        super(RandomWeightedAverage, self).__init__()
        self.batch_size = batch_size
        self.predict_terms = predict_terms
    def _merge_function(self, inputs):
        alpha = K.random_uniform((self.batch_size, self.predict_terms, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self, args, encoder, cpc):
        self.img_rows = args.image_size
        self.img_cols = args.image_size
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = args.code_size
        self.predict_terms = args.predict_terms

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=(self.predict_terms, self.img_rows, self.img_cols, self.channels))

        # Noise input
        z_disc = Input(shape=(self.predict_terms, self.latent_dim))
        pred = Input(shape=(self.predict_terms, self.latent_dim))

        z_disc_con = keras.layers.Lambda(lambda i: K.concatenate([i[0], i[1]], axis=-1))([z_disc, pred])

        # Generate image based of noise (fake sample)

        fake_img = keras.layers.Concatenate(axis=1)([keras.layers.Lambda(lambda z: K.expand_dims(z, 1))(self.generator(keras.layers.Lambda(lambda z: z[:,0])(z_disc_con))) for i in range(self.predict_terms)])

        # Discriminator determines validity of the real and fake images

        fake = keras.layers.Concatenate(axis=1)([keras.layers.Lambda(lambda z: K.expand_dims(z, 1))(self.critic(keras.layers.Lambda(lambda z: z[:,0])(fake_img))) for i in range(self.predict_terms)])
        valid = keras.layers.Concatenate(axis=1)([keras.layers.Lambda(lambda z: K.expand_dims(z, 1))(self.critic(keras.layers.Lambda(lambda z: z[:,0])(real_img))) for i in range(self.predict_terms)])

        # Construct weighted average between real and fake images

        interpolated_img = RandomWeightedAverage(args.batch_size, args.predict_terms)([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = keras.layers.Concatenate(axis=1)([keras.layers.Lambda(lambda z: K.expand_dims(z, 1))(self.critic(keras.layers.Lambda(lambda z: z[:,0])(interpolated_img))) for i in range(self.predict_terms)])

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc, pred],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        # self.critic_model.summary()
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.predict_terms, self.latent_dim))
        pred = Input(shape=(self.predict_terms, self.latent_dim))


        z_gen_con = keras.layers.Lambda(lambda i: K.concatenate([i[0], i[1]], axis=-1))([z_gen, pred])

        # Generate images based of noise
        img = keras.layers.Concatenate(axis=1)([keras.layers.Lambda(lambda z: K.expand_dims(z, 1))(self.generator(keras.layers.Lambda(lambda z: z[:,i])(z_gen_con))) for i in range(self.predict_terms)])
        # Discriminator determines validity
        valid = keras.layers.Concatenate(axis=1)([keras.layers.Lambda(lambda z: K.expand_dims(z, 1))(self.critic(keras.layers.Lambda(lambda z: z[:,0])(img))) for i in range(self.predict_terms)])
        # Defines generator model
        z = keras.layers.Concatenate(axis=1)([keras.layers.Lambda(lambda z: K.expand_dims(z, 1))(encoder(keras.layers.Lambda(lambda z: z[:,0])(img))) for i in range(self.predict_terms)])

        # tz = keras.layers.Lambda(lambda z: K.expand_dims(z, 1))(z)
        # tpred = keras.layers.Lambda(lambda z: K.expand_dims(z, 1))(pred)
        cpc_loss = cpc([pred, z])

        self.generator_model = Model(inputs=[z_gen, pred], outputs=[valid, cpc_loss, img])
        self.generator_model.compile(loss=[self.wasserstein_loss, 'binary_crossentropy', None], loss_weights=[args.gan_weight, 1.0, 0.0], optimizer=optimizer)
        # self.generator_model.summary()

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]

        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(2, len(gradients_sqr.shape)))

        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim * 2))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        # model.summary()

        noise = Input(shape=(self.latent_dim * 2,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        # model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)





def network_encoder(x, code_size):

    ''' Define the network mapping images to embeddings '''

    # x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.LeakyReLU()(x)
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

def network_autoregressive(x):

    ''' Define the network that integrates information along the sequence '''

    x = keras.layers.GRU(units=256, return_sequences=False, name='ar_context')(x)

    return x


def network_prediction(context, code_size, predict_terms, name='z'):

    ''' Define the network mapping context to multiple embeddings '''

    outputs = []
    for i in range(predict_terms):
        outputs.append(keras.layers.Dense(units=code_size, activation="linear", name=name+'_t_{i}'.format(i=i))(context))

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

        # dot_product = K.mean(20 - K.abs(y_encoded  - preds) * (y_encoded - preds), axis=-1)
        # dot_product = K.mean(K.l2_normalize(y_encoded, axis=-1) * K.l2_normalize(preds, axis=-1), axis=-1)
        dot_product = K.mean(y_encoded * preds, axis=-1)
        dot_product = K.mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension
    
        # Keras loss functions take probabilities
        dot_product_probs = K.sigmoid(dot_product)

        return dot_product_probs

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
    # encoder_model.summary()

    # Define rest of model
    x_input = keras.layers.Input((terms, image_shape[0], image_shape[1], image_shape[2]))
    x_encoded = keras.layers.TimeDistributed(encoder_model)(x_input)

    context = network_autoregressive(x_encoded)
    preds = network_prediction(context, code_size, predict_terms)

    y_input = keras.layers.Input((predict_terms, image_shape[0], image_shape[1], image_shape[2]))
    y_encoded = keras.layers.TimeDistributed(encoder_model)(y_input)

    # Loss
    cpc_layer = CPCLayer()
    dot_product_probs = cpc_layer([preds, y_encoded])


    # Model
    cpc_model = keras.models.Model(inputs=[x_input, y_input], outputs=[preds, dot_product_probs])


    # Compile model
    cpc_model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss=[None, 'binary_crossentropy'],
        metrics=['binary_accuracy']
    )
    # cpc_model.summary()


    encoder_model.trainable = False
    cpc_layer.trainable = False

    return (cpc_model, encoder_model, cpc_layer)


def train_model(args, batch_size, output_dir, code_size, lr=1e-4, terms=4, predict_terms=4, image_size=28, color=False):

    # Prepare data
    train_data = SortedNumberGenerator(batch_size=batch_size, subset='train', terms=terms,
                                       positive_samples=batch_size // 2, predict_terms=predict_terms,
                                       image_size=image_size, color=color, rescale=True)

    validation_data = SortedNumberGenerator(batch_size=batch_size, subset='valid', terms=terms,
                                            positive_samples=batch_size // 2, predict_terms=predict_terms,
                                            image_size=image_size, color=color, rescale=True)


    model, encoder, cpc = network_cpc(args, image_shape=(image_size, image_size, 3), terms=terms, predict_terms=predict_terms,
                        code_size=code_size, learning_rate=lr)
    gan = WGANGP(args, encoder, cpc)

    if len(args.load_name) > 0:
        model = keras.models.load_model(join(output_dir, args.load_name))

    else:
        print('Start Training CPC')
        for epoch in range(args.cpc_epochs // 1):

            avg0, avg1, avg2, avg3 = [], [], [], []
            for i in range(len(train_data) // 1):
                train_batch = next(train_data)
                train_result = model.train_on_batch(train_batch[0][:2], train_batch[1])
                avg0.append(train_result[0])
                avg2.append(train_result[2])
                sys.stdout.write(
                    '\r Epoch {}: training[{} / {}]'.format(epoch, i, len(train_data)))

            for i in range(len(validation_data) // 1):
                validation_batch = next(validation_data)
                validation_result = model.test_on_batch(validation_batch[0][:2], validation_batch[1])
                avg1.append(validation_result[0])
                avg3.append(validation_result[2])
                sys.stdout.write(
                    '\r Epoch {}: validation[{} / {}]'.format(epoch, i, len(validation_data)))

            print('\n%s' % ('-' * 40))
            print('Train loss: %.2f, Accuracy: %.2f \t Validation loss: %.2f, Accuracy: %.2f' % (100.0 * np.mean(avg0), 100.0 * np.mean(avg2), 100.0 * np.mean(avg1), 100.0 * np.mean(avg3)))
            print('%s' % ('-' * 40))

        

        # Saves the model
        # Remember to add custom_objects={'CPCLayer': CPCLayer} to load_model when loading from disk
        model.save(join(output_dir, 'cpc_' + args.name + '.h5'))

        # Saves the encoder alone
        encoder = model.layers[1].layer
        encoder.save(join(output_dir, 'encoder_' + args.name + '.h5'))

    print('\nStart Training GAN')
    # Adversarial ground truths
    valid = -np.ones((batch_size, predict_terms, 1))
    fake =  np.ones((batch_size, predict_terms, 1))
    cpc_true = np.ones((batch_size, 1))
    dummy = np.zeros((batch_size, predict_terms, 1)) # Dummy gt for gradient penalty

    # self.critic_model = Model(inputs=[real_img, z_disc, pred],
    #                     outputs=[valid, fake, validity_interpolated])
    # self.generator_model = Model(inputs=[z_gen, pred], outputs=[valid, cpc_loss, img])

    for epoch in range(args.gan_epochs):


        avg0, avg1, avg2 = [], [], []
        for i in range(len(train_data) // 1):
            train_batch = next(train_data)

            preds, _ = model.predict(train_batch[0][:2], batch_size=batch_size)

            for _ in range(5):
                # r1 = [random.randint(0,1) for __ in range(predict_terms)]
                r2 = [random.randint(0,terms - 1) for __ in range(predict_terms)]
                images = []
                for j in range(predict_terms):
                    images.append(train_batch[0][0][:, r2[j]])
                image = np.array(images).transpose(1, 0, 2, 3, 4)
                noise = np.random.normal(0, 1, (batch_size, predict_terms, args.code_size))
                d_loss = gan.critic_model.train_on_batch([image, noise, preds], [valid, fake, dummy])
                avg2.append(d_loss[0])


            image = train_batch[0][2]
            g_loss = gan.generator_model.train_on_batch([noise, preds], [valid, cpc_true])
            avg0.append(g_loss[1])
            avg1.append(g_loss[2])

            _, _, recon = gan.generator_model.predict([noise, preds], batch_size=batch_size)
            sys.stdout.write(
                '\r Epoch {}: train[{} / {}]'.format(epoch, i, len(train_data)))

        print('\n%s' % ('-' * 40))
        print('Training -- Generator Critic loss: %.2f, Generator CPC loss: %.2f, Discriminator: %.2f' % (100.0 * np.mean(avg0), 100.0 * np.mean(avg1), 100.0 * np.mean(avg2)))


        avg0, avg1, avg2 = [], [], []

        for i in range(max(1, len(validation_data) // 1)):
            validation_batch = next(validation_data)

            preds, _ = model.predict(validation_batch[0][:2], batch_size=batch_size)
            noise = np.random.normal(0, 1, (batch_size, predict_terms, args.code_size))
            image = validation_batch[0][1]
            d_loss = gan.critic_model.test_on_batch([image, noise, preds], [valid, fake, dummy])
            avg2.append(d_loss[0])
            image = validation_batch[0][2]
            g_loss = gan.generator_model.test_on_batch([noise, preds], [valid, cpc_true])
            _, _, recon = gan.generator_model.predict([noise, preds], batch_size=batch_size)
            avg0.append(g_loss[1])
            avg1.append(g_loss[2])

            sys.stdout.write(
                '\r Epoch {}: validation[{} / {}]'.format(epoch, i, len(validation_data)))


        print('\n')
        print('Validation -- Generator Critic loss: %.2f, Generator CPC loss: %.2f, Discriminator: %.2f' % (100.0 * np.mean(avg0), 100.0 * np.mean(avg1), 100.0 * np.mean(avg2)))
        print('%s' % ('-' * 40))

        if epoch % 1 == 0:
            fig = plt.figure()
            for i in range(terms):
                ax = fig.add_subplot(2, terms + predict_terms, i+1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(validation_batch[0][0][0][i] * 0.5 + 0.5)
        
            for i in range(predict_terms):
                ax = fig.add_subplot(2, terms + predict_terms, terms+i+1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(validation_batch[0][2][0][i] * 0.5 + 0.5)

            for i in range(terms):
                ax = fig.add_subplot(2, terms + predict_terms, terms + predict_terms + i+1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(validation_batch[0][0][0][i] * 0.5 + 0.5)

            for i in range(predict_terms):
                ax = fig.add_subplot(2, terms + predict_terms, terms + predict_terms + terms+i+1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(recon[0][i] * 0.5 + 0.5)
            plt.axis('off')
            plt.savefig('fig/' + args.name + '_epoch' + str(epoch) + '.png')
    
    # Saves the model
    # Remember to add custom_objects={'CPCLayer': CPCLayer} to load_model when loading from disk
    gan.generator_model.save(join(output_dir, 'generator_' + args.name + '.h5'))

    # Saves the encoder alone
    gan.critic_model.save(join(output_dir, 'dis_' + args.name + '.h5'))

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description='CPC')
    argparser.add_argument(
        '--name',
        default='cpc',
        help='name')
    argparser.add_argument(
        '--load-name',
        default='',
        help='loadpath')
    argparser.add_argument(
        '-e', '--cpc-epochs',
        default=30,
        type=int,
        help='cpc epochs')
    argparser.add_argument(
        '-g', '--gan-epochs',
        default=1000000,
        type=int,
        help='gan epochs')
    argparser.add_argument(
        '--lr',
        default=1e-3,
        type=float,
        help='Learning rate')
    argparser.add_argument(
        '--gan-weight',
        default=0.01,
        type=float,
        help='GAN Weight')
    argparser.add_argument(
        '--code-size',
        default=32,
        type=int,
        help='Code Size')
    argparser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='Batch Size')
    argparser.add_argument(
        '--predict-terms',
        default=10,
        type=int,
        help='Predict Terms')
    argparser.add_argument(
        '--terms',
        default=10,
        type=int,
        help='X Terms')
    argparser.add_argument(
        '--image-size',
        default=28,
        type=int,
        help='Image size')
        
    args = argparser.parse_args()

    train_model(
        args, 
        batch_size=args.batch_size,
        output_dir='models/',
        code_size=args.code_size,
        lr=args.lr,
        terms=args.terms,
        predict_terms=args.predict_terms,
        image_size=args.image_size,
        color=False
    ) 

