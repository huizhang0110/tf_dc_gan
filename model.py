import os
import tensorflow as tf
from tensorflow import logging, gfile
import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
output_dir = "./local_run"
if not gfile.Exists(output_dir):
    gfile.MakeDirs(output_dir)


def get_default_params():
    return tf.contrib.training.HParams(
        z_dim=100,
        init_conv_size=4,
        g_channels=[128, 64, 32, 1],
        d_channels=[32, 64, 128, 256],
        batch_size=128,
        learning_rate=0.002,
        beta1=0.5,
        img_size=32,
    )


hps = get_default_params()
print(hps)


class MnistData:

    def __init__(self, x_train, z_dim, img_size):
        self._data = x_train
        self._example_num = len(self._data)
        self._z_data = np.random.standard_normal(size=(self._example_num, z_dim))
        self._indicator = 0
        self._resize_mnist_img(img_size)
        self._random_shuffle()

    def _random_shuffle(self):
        p = np.random.permutation(self._example_num)
        self._z_data = self._z_data[p]
        self._data = self._data[p]

    def _resize_mnist_img(self, img_size):
        """Resize mnist image to goal img_size
        numpy --> PIL img
        PIL img -> resize image --> numpy
        """
        data = np.asarray(self._data * 255, np.uint8).reshape((self._example_num, 28, 28, 1))
        new_data = [] 
        for i in range(self._example_num):
            img = data[i].reshape((28, 28))
            img = Image.fromarray(img)
            img = img.resize((img_size, img_size))
            img = np.asarray(img)
            img = img.reshape((img_size, img_size, 1))
            new_data.append(img)
        new_data = np.asarray(new_data, dtype=np.float32)
        new_data = new_data / 127.5 - 1  # [-1, 1]
        self._data = new_data

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > self._example_num:
            self._random_shuffle()
            self._indicator = 0
            end_indicator = self._indicator + batch_size
        assert end_indicator < self._example_num
        batch_data = self._data[self._indicator:end_indicator]
        batch_z = self._z_data[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_z


mnist_data = MnistData(x_train, hps.z_dim, hps.img_size)


# Build Graph
def conv2d(inputs, out_channel, name, training):
    with tf.variable_scope(name):
        conv2d_output = tf.layers.conv2d(inputs, out_channel, [5, 5], strides=[2, 2], padding="SAME")
        bn = tf.layers.batch_normalization(conv2d_output, training=training)
        return tf.nn.leaky_relu(bn)


def conv2d_transpose(inputs, out_channel, name, is_training, with_bn_relu=True):
    """Wrapper of conv2d transpose"""
    with tf.variable_scope(name):
        conv2d_trans = tf.layers.conv2d_transpose(inputs, out_channel, [5, 5], strides=[2, 2], padding="SAME")
        if with_bn_relu:
            bn = tf.layers.batch_normalization(conv2d_trans, training=is_training)
            return tf.nn.relu(bn)
        else:
            return conv2d_trans


class Generator:

    def __init__(self, channels, init_conv_size):
        self._channels = channels
        self._init_conv_size = init_conv_size
        self._reuse = False

    def __call__(self, inputs, training):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope("Generator", reuse=self._reuse):
            """random_vector -> fc -> self._channels[0] * init_conv_size ** 2
            --> reshape [_init_conv_size, _init_conv_size, channels[0]]
            """
            fc = tf.layers.dense(inputs, self._channels[0] *self._init_conv_size * self._init_conv_size)
            conv0 = tf.reshape(fc, [-1, self._init_conv_size, self._init_conv_size, self._channels[0]])
            bn0 = tf.layers.batch_normalization(conv0, training=training)
            relu0 = tf.nn.relu(bn0)

            deconv_inputs = relu0
            for i in range(1, len(self._channels)):
                with_bn_relu = (i != len(self._channels) - 1)
                deconv_inputs = conv2d_transpose(
                    deconv_inputs,
                    self._channels[i],
                    "deconv-%d" % i,
                    training,
                    with_bn_relu
                )
            img_inputs = deconv_inputs
            with tf.variable_scope("generate_images"):
                imgs = tf.tanh(img_inputs, name="img")

            self._reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator") 
        return imgs


class Discriminator:

    def __init__(self, channels):
        self._channels = channels
        self._reuse = False

    def __call__(self, inputs, training):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        conv_inputs = inputs
        with tf.variable_scope("Discriminator", reuse=self._reuse):
            for i in range(len(self._channels)):
                conv_inputs = conv2d(conv_inputs, self._channels[i], "conv-%d" % i, training)
            fc_inputs = conv_inputs
            with tf.variable_scope("fc"):
                flatten = tf.layers.flatten(fc_inputs)
                logits = tf.layers.dense(flatten, 2, name="logits")
            self._reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")
        return logits



class DCGAN:

    def __init__(self, hps):
        g_channels = hps.g_channels
        d_channels = hps.d_channels
        self._batch_size = hps.batch_size
        self._init_conv_size = hps.init_conv_size
        self._z_dim = hps.z_dim
        self._img_size = hps.img_size

        self._generator = Generator(g_channels, self._init_conv_size)
        self._discriminator = Discriminator(d_channels)

    def build(self):
        self._z_placeholder = tf.placeholder(
            tf.float32, [self._batch_size, self._z_dim])
        self._img_placeholder = tf.placeholder(
            tf.float32, [self._batch_size, self._img_size, self._img_size, 1])

        generated_imgs = self._generator(self._z_placeholder, training=True)
        fake_img_logits = self._discriminator(generated_imgs, training=True)
        real_img_logits = self._discriminator(self._img_placeholder, training=True)

        loss_on_fake_to_real = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.ones([self._batch_size], dtype=tf.int64),
                logits=fake_img_logits
            )
        )  # 生成器主要使要生成的图片成真实

        loss_on_fake_to_fake = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.zeros([self._batch_size], dtype=tf.int64), 
                logits=fake_img_logits
            )
        )
        loss_on_real_to_real = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.ones([self._batch_size], dtype=tf.int64),
                logits=real_img_logits
            )
        )

        tf.add_to_collection("g_losses", loss_on_fake_to_real)
        tf.add_to_collection("d_losses", loss_on_fake_to_fake)
        tf.add_to_collection("d_losses", loss_on_real_to_real)

        loss = {
            "g": tf.add_n(tf.get_collection("g_losses"), name="total_g_loss"),
            "d": tf.add_n(tf.get_collection("d_losses"), name="total_d_loss")
        }

        return self._z_placeholder, self._img_placeholder, generated_imgs, loss

    def build_train_op(self, losses, learning_rate, beta1):
        """Build train op, should be called after build is called"""
        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        g_opt_op = g_opt.minimize(losses["g"], var_list=self._generator.variables)
        d_opt_op = g_opt.minimize(losses["d"], var_list=self._discriminator.variables)

        with tf.control_dependencies([g_opt_op, d_opt_op]):
            return tf.no_op(name="train")



dc_gan = DCGAN(hps)
z_placeholder, img_placeholder, generated_imgs, losses = dc_gan.build()
train_op = dc_gan.build_train_op(losses, hps.learning_rate, hps.beta1)

# Training process
def combine_imgs(batch_imgs, img_size, rows=8, cols=8):
    """Combines small images in a batch into a big pic
    [batch_size, img_size, img_size, 1]
    """
    result_big_img = []
    for i in range(rows):
        row_imgs = []
        for j in range(cols):
            img = batch_imgs[cols * i + j]
            img = img.reshape((img_size, img_size))
            img = (img + 1) * 127.5
            row_imgs.append(img)
        row_imgs = np.hstack(row_imgs)
        result_big_img.append(row_imgs)
    result_big_img = np.vstack(result_big_img)
    result_big_img = np.asarray(result_big_img, np.uint8)
    result_big_img = Image.fromarray(result_big_img)
    return result_big_img


train_steps = 100000
with tf.Session() as sess:
    sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])
    logging.info("Start training")
    for step in range(train_steps):
        batch_imgs, batch_z = mnist_data.next_batch(hps.batch_size)
        fetches = [train_op, losses["g"], losses["d"]]
        should_sample = (step + 1) % 1000 == 0
        if should_sample:
            fetches += [generated_imgs]

        output_values = sess.run(fetches, feed_dict={
            z_placeholder: batch_z,
            img_placeholder: batch_imgs
        })

        if should_sample:
            _, g_loss_val, d_loss_val = output_values[:3]
            logging.info("step %d, g_loss: %4.3f d_loss: %4.3f" % (step, g_loss_val, d_loss_val))

            gen_imgs_val = output_values[3]
            gen_img_path = os.path.join(output_dir, "%05d-gen.jpg" % (step + 1, ))
            gt_img_path = os.path.join(output_dir, "%05d-gt.jpg" % (step + 1, ))

            gen_img = combine_imgs(gen_imgs_val, hps.img_size)
            gt_img = combine_imgs(batch_imgs, hps.img_size)

            gen_img.save(gen_img_path)
            gt_img.save(gt_img_path)

