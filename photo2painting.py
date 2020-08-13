import tensorflow as tf
import argparse
import utils
import warnings
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Convert image to Van Gogh painting.')

parser.add_argument('-image',
                    dest='image',
                    default='./bath_photo.jpg',
                    help='File path of the image you want changed.')

args = parser.parse_args()

generator = utils.generator()

generator.load_weights('./checkpoints/generator_f')

#load image for generator
output_image = tf.keras.preprocessing.image.load_img(args.image)
image = tf.keras.preprocessing.image.img_to_array(output_image)
image = tf.keras.preprocessing.image.smart_resize(image,
                                                  [256, 256])
input_image = utils.normalize(image)

x, y, z = input_image.shape

generated_image = generator(tf.reshape(input_image, shape=[1, x, y, z]))


fig, ax = plt.subplots(1, 2, figsize=(10,5))

ax[0].imshow(output_image)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(generated_image[0] * 0.5 + 0.5)
ax[1].set_title('Generated Painting')
ax[1].axis('off')

plt.show()
