import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import tensorflow_addons as tfa


def encoder_block(filters, size, batch_norm=True):

    '''Encoding block
       --------------
       filters : The dimensionality of the output space
       size : Size of the kernel along convolutional layer
       batch_norm : Apply batch normalization'''

    init = tf.random_normal_initializer(0.0, 0.02)

    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2D(filters,
                                     size,
                                     strides=2,
                                     padding='same',
                                     kernel_initializer=init,
                                     use_bias=False))
    if batch_norm:
        block.add(tfa.layers.InstanceNormalization())

    block.add(tf.keras.layers.LeakyReLU())

    return block

def decoder_block(filters, size, dropout=False):

    init = tf.random_normal_initializer(0.0, 0.02)

    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2DTranspose(filters,
                                              size,
                                              strides=2,
                                              padding='same',
                                              kernel_initializer=init,
                                              use_bias=False))
    block.add(tfa.layers.InstanceNormalization())

    if dropout:

        block.add(tf.keras.layers.Dropout(0.5))

    block.add(tf.keras.layers.LeakyReLU())

    return block

def generator():

    '''Generator model for the GAN'''

    inputs = tf.keras.layers.Input(shape=[None, None, 3])

    encoder_stack = [encoder_block(64, 4, batch_norm=False),
                     encoder_block(128, 4),
                     encoder_block(256, 4),
                     encoder_block(512, 4),
                     encoder_block(512, 4),
                     encoder_block(512, 4),
                     encoder_block(512, 4),
                     encoder_block(512, 4)]

    decoder_stack = [decoder_block(512, 4, dropout=True),
                     decoder_block(512, 4, dropout=True),
                     decoder_block(512, 4, dropout=True),
                     decoder_block(512, 4),
                     decoder_block(256, 4),
                     decoder_block(128, 4),
                     decoder_block(64, 4)]

    init = tf.random_normal_initializer(0.0, 0.02)

    last = tf.keras.layers.Conv2DTranspose(3,
                                           4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=init,
                                           activation='tanh')
    concat = tf.keras.layers.Concatenate()

    x = inputs

    skips = []

    for down in encoder_stack:

        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(decoder_stack, skips):

        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def discriminator():

    '''Discriminator Model for our GAN.'''

    init = tf.random_normal_initializer(0.0, 0.02)

    inputs = tf.keras.layers.Input(shape=[256, 256, 3],
                                   name='input_images')


    down1 = encoder_block(64, 4, batch_norm=False)(inputs)
    down2 = encoder_block(128, 4)(down1)
    down3 = encoder_block(256, 4)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512,
                                  4,
                                  strides=1,
                                  kernel_initializer=init,
                                  use_bias=False)(zero_pad1)

    batch_norm1 = tfa.layers.InstanceNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batch_norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    last = tf.keras.layers.Conv2D(1,
                                  4,
                                  strides=1,
                                  kernel_initializer=init)(zero_pad2)

    return tf.keras.Model(inputs=inputs, outputs=last)

def normalize(image):

    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1

    return image
