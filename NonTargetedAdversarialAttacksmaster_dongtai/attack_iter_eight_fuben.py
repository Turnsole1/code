"""实现8个模型给出对抗样本的标签 """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave

import tensorflow as tf
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

import time

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v3', './models/inception_v3.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_adv_inception_v3', './models/adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens3_adv_inception_v3', './models/ens3_adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens4_adv_inception_v3', './models/ens4_adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v4', './models/inception_v4.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_resnet_v2', './models/inception_resnet_v2_2016_08_30.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens_adv_inception_resnet_v2', './models/ens_adv_inception_resnet_v2_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_resnet', './models/resnet_v2_101.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', './adv_samples', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', './out_images_v3_fuben', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')  #the begin is 16.0

tf.flags.DEFINE_integer(
    'num_iter', 40, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 2, 'How many images process at one time.')

tf.flags.DEFINE_float(
    'momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_string(
    'GPU_ID', '3', 'which GPU to use.')

tf.flags.DEFINE_integer(
    'LOW_K', 5, 'Get the smallest number of the first k..')

FLAGS = tf.flags.FLAGS

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.GPU_ID

def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath,'rb') as f:
      image = imread(f, mode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
      images: array with minibatch of images
      filenames: list of filenames without path
        If number of file names in this list less than number of images in
        the minibatch then only first len(filenames) images will be saved.
      output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
      # Images for inception classifier are normalized to be in [-1, 1] interval,
      # so rescale them back to [0, 1].
      with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
        imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


# def graph(x, y, i, x_max, x_min, grad, pred):
#   eps = 2.0 * FLAGS.max_epsilon / 255.0
#   num_iter = FLAGS.num_iter
#   alpha = eps / num_iter
#   momentum = FLAGS.momentum
#   num_classes = 1001
#
#   with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
#     logits_v3, end_points_v3 = inception_v3.inception_v3(
#         x, num_classes=num_classes, is_training=False)
#
#   with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
#     logits_adv_v3, end_points_adv_v3 = inception_v3.inception_v3(
#         x, num_classes=num_classes, is_training=False, scope='AdvInceptionV3')
#
#   with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
#     logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
#         x, num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')
#
#   with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
#     logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
#         x, num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')
#
#   with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
#     logits_v4, end_points_v4 = inception_v4.inception_v4(
#         x, num_classes=num_classes, is_training=False)
#
#   with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
#     logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
#         x, num_classes=num_classes, is_training=False)
#
#   with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
#     logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
#         x, num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')
#
#   with slim.arg_scope(resnet_v2.resnet_arg_scope()):
#     logits_resnet, end_points_resnet = resnet_v2.resnet_v2_101(
#         x, num_classes=num_classes, is_training=False)
#
#   pred = tf.argmax(end_points_v3['Predictions'] + end_points_adv_v3['Predictions'] + end_points_ens3_adv_v3['Predictions'] + \
#                   end_points_ens4_adv_v3['Predictions'] + end_points_v4['Predictions'] + \
#                   end_points_res_v2['Predictions'] + end_points_ensadv_res_v2['Predictions'] + end_points_resnet['predictions'], 1)
#
#   logits = tf.constant(0, dtype = tf.float32, shape=[FLAGS.batch_size, 1001])
#   # !!!!!!!!!!!!!!!!!! maybe has problem !!!!!!!!!!!!
#   logits_dict = [logits_v3, logits_ens3_adv_v3, logits_ens4_adv_v3, logits_v4, logits_res_v2, logits_ensadv_res_v2, logits_resnet]
#   for i in range(FLAGS.batch_size):
#       end_points_v3_Pred = end_points_v3['Predictions'][i][pred[i]]
#       # end_points_adv_v3_Pred = end_points_adv_v3['Predictions']
#       end_points_ens3_adv_v3_Pred = end_points_ens3_adv_v3['Predictions'][i][pred[i]]
#       end_points_ens4_adv_v3_Pred = end_points_ens4_adv_v3['Predictions'][i][pred[i]]
#       end_points_v4_Pred = end_points_v4['Predictions'][i][pred[i]]
#       end_points_res_v2_Pred = end_points_res_v2['Predictions'][i][pred[i]]
#       end_points_ensadv_res_v2_Pred = end_points_ensadv_res_v2['Predictions'][i][pred[i]]
#       end_points_resnet_Pred = end_points_resnet['predictions'][i][pred[i]]
#
#       ens_Pred_Value = [end_points_v3_Pred * -1, end_points_ens3_adv_v3_Pred * -1, end_points_ens4_adv_v3_Pred * -1,
#                         end_points_v4_Pred * -1, end_points_res_v2_Pred * -1, end_points_ensadv_res_v2_Pred * -1,
#                         end_points_resnet_Pred * -1]
#       TopKFit, TopKFitIndx = tf.nn.top_k(ens_Pred_Value, FLAGS.LOW_K)
#       # TopKFit = TopKFit * -1
#       logitsi = tf.placeholder()
#       logits[i] = (logits_dict[TopKFitIndx[0]] * 5 + logits_dict[TopKFitIndx[1]] * 4 + logits_dict[TopKFitIndx[2]] * 3 + logits_dict[TopKFitIndx[3]] * 2 + logits_dict[TopKFitIndx[4]] * 1) / 15
#
#   first_round = tf.cast(tf.equal(i, 0), tf.int64)
#   y = first_round * pred + (1 - first_round) * y
#   one_hot = tf.one_hot(y, num_classes)
#   # if (tf.less(end_points_v3['Predictions'][0][pred[0]], tf.constant(0.6))):
#   #     list.append(end_points_v3['Predictions'][0][pred[0]])
#   # if (tf.less(end_points_ens3_adv_v3['Predictions'][0][pred[0]], 0.6)):
#   #     list.append(end_points_ens3_adv_v3['Predictions'][0][pred[0]])
#   # if(tf.less(end_points_ens4_adv_v3['Predictions'][0][pred[0]], 0.6)):
#   #     list.append(end_points_ens4_adv_v3['Predictions'][0][pred[0]])
#   # if (tf.less(end_points_v4['Predictions'][0][pred[0]], 0.6)):
#   #     list.append(end_points_v4['Predictions'][0][pred[0]])
#   # if (tf.less(end_points_res_v2['Predictions'][0][pred[0]], 0.6)):
#   #     list.append(end_points_res_v2['Predictions'][0][pred[0]])
#   # if(tf.less(end_points_ensadv_res_v2['Predictions'][0][pred[0]], 0.6)):
#   #     list.append(end_points_ensadv_res_v2['Predictions'][0][pred[0]])
#   # if (tf.less(end_points_resnet['Predictions'][0][pred[0]], 0.6)):
#   #     list.append(end_points_resnet['Predictions'][0][pred[0]])
#   # logits = (logits_v3 + 0.25 * logits_adv_v3 + logits_ens3_adv_v3 + \
#   #          logits_ens4_adv_v3 + logits_v4 + \
#   #          logits_res_v2 + logits_ensadv_res_v2 + logits_resnet) / 7.25
#   # auxlogits = (end_points_v3['AuxLogits'] + 0.25 * end_points_adv_v3['AuxLogits'] + end_points_ens3_adv_v3['AuxLogits'] + \
#   #             end_points_ens4_adv_v3['AuxLogits'] + end_points_v4['AuxLogits'] + \
#   #             end_points_res_v2['AuxLogits'] + end_points_ensadv_res_v2['AuxLogits']) / 6.25
#   cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
#                                                   logits,
#                                                   label_smoothing=0.0,
#                                                   weights=1.0)
#   # cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
#   #                                                  auxlogits,
#   #                                                  label_smoothing=0.0,
#   #                                                  weights=0.4)
#
#   noise = tf.gradients(cross_entropy, x)[0]
#   noise = noise / tf.reduce_mean(tf.abs(noise), [1,2,3], keep_dims=True)
#   noise = momentum * grad + noise
#   x = x + alpha * tf.sign(noise)
#   x = tf.clip_by_value(x, x_min, x_max)
#   i = tf.add(i, 1)
#   return x, y, i, x_max, x_min, noise, pred

# def stop(x, y, i, x_max, x_min, grad, pred):
#   num_iter = FLAGS.num_iter
#   return tf.less(i, num_iter)


def main(_):
    start = time.clock()
    # tf.set_random_seed(1)
    # np.random.seed(1)


    # Images for inception classifier are normalized to be in [-1, 1] interval,3
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    tf.logging.set_verbosity(tf.logging.INFO)

    x = tf.placeholder(dtype=tf.float32, shape=batch_shape) # , name='x_Placeholder'

    x_max = tf.clip_by_value(x + eps, -1.0, 1.0)
    x_min = tf.clip_by_value(x - eps, -1.0, 1.0)

    # y = tf.zeros(shape=FLAGS.batch_size, dtype=tf.int64)
    # i = tf.zeros(shape=1, dtype=tf.int8)
    grad = tf.zeros(shape=batch_shape)
    # pred = tf.zeros(shape=FLAGS.batch_size, dtype=tf.int64)
    # noise = tf.zeros(shape=[FLAGS.batch_size, 299, 299, 3],dtype=tf.float32)
    # cross_entropy = tf.zeros(shape=FLAGS.batch_size,dtype=tf.float32)



    with tf.Session() as sess:
        eps = 2.0 * FLAGS.max_epsilon / 255.0
        num_iter = FLAGS.num_iter
        alpha = eps / num_iter
        momentum = FLAGS.momentum
        num_classes = 1001

        # pred = tf.zeros(shape=FLAGS.batch_size, dtype=tf.int64)
        # noise = tf.zeros(shape=[FLAGS.batch_size, 299, 299, 3], dtype=tf.float32)
        # cross_entropy = tf.zeros(shape=FLAGS.batch_size, dtype=tf.float32)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_v3, end_points_v3 = inception_v3.inception_v3(
                x, num_classes=num_classes, is_training=False)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_adv_v3, end_points_adv_v3 = inception_v3.inception_v3(
                x, num_classes=num_classes, is_training=False, scope='AdvInceptionV3')

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
                x, num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
                x, num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')

        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits_v4, end_points_v4 = inception_v4.inception_v4(
                x, num_classes=num_classes, is_training=False)

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
                x, num_classes=num_classes, is_training=False)

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
                x, num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')

        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits_resnet, end_points_resnet = resnet_v2.resnet_v2_101(
                x, num_classes=num_classes, is_training=False)

        pred_dict = end_points_v3['Predictions'] + end_points_adv_v3['Predictions'] + end_points_ens3_adv_v3['Predictions'] + \
            end_points_ens4_adv_v3['Predictions'] + end_points_v4['Predictions'] + \
            end_points_res_v2['Predictions'] + end_points_ensadv_res_v2['Predictions'] + end_points_resnet['predictions']

        pred = tf.argmax(pred_dict, axis = 1)

        logits_dict = [logits_v3, logits_ens3_adv_v3, logits_ens4_adv_v3, logits_v4, logits_res_v2,
                       logits_ensadv_res_v2,
                       logits_resnet]
        # for j in range(FLAGS.batch_size):
        #     end_points_v3_Pred = end_points_v3['Predictions'][j][pred[j]]
        #     # end_points_adv_v3_Pred = end_points_adv_v3['Predictions']
        #     end_points_ens3_adv_v3_Pred = end_points_ens3_adv_v3['Predictions'][j][pred[j]]
        #     end_points_ens4_adv_v3_Pred = end_points_ens4_adv_v3['Predictions'][j][pred[j]]
        #     end_points_v4_Pred = end_points_v4['Predictions'][j][pred[j]]
        #     end_points_res_v2_Pred = end_points_res_v2['Predictions'][j][pred[j]]
        #     end_points_ensadv_res_v2_Pred = end_points_ensadv_res_v2['Predictions'][j][pred[j]]
        #     end_points_resnet_Pred = end_points_resnet['Predictions'][j][pred[j]]
        #
        #
        #     ens_Pred_Value = [end_points_v3_Pred * -1, end_points_ens3_adv_v3_Pred * -1,
        #                       end_points_ens4_adv_v3_Pred * -1,
        #                       end_points_v4_Pred * -1, end_points_res_v2_Pred * -1,
        #                       end_points_ensadv_res_v2_Pred * -1,
        #                       end_points_resnet_Pred * -1]
        #     TopKFit, TopKFitIndx = tf.nn.top_k(ens_Pred_Value, FLAGS.LOW_K)
        #     # TopKFit = TopKFit * -1
        #
        #     logits_dictionary = tf.placeholder(dtype=tf.int32, shape=(7, FLAGS.batch_size, 1001))
        #     TopKFitIndex = tf.placeholder(dtype=tf.int32, shape=(FLAGS.batch_size))
        #     logits[j] = (logits_dictionary[TopKFitIndex[0]] * 5.0 + logits_dictionary[TopKFitIndex[1]] * 4.0 + logits_dictionary[TopKFitIndex[2]] * 3.0 + logits_dictionary[TopKFitIndex[3]] * 2.0 + logits_dictionary[TopKFitIndex[4]] * 1.0) / 15.0

        #i_iteration = tf.placeholder(dtype=tf.int32)
        #first_round = tf.cast(tf.equal(i_iteration, 0), tf.int64)
        #y = first_round * pred
        y = tf.placeholder(dtype=tf.int32, shape=[2,])                 # shape= FLAGS.batchsize !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #y1 = first_round * pred + (1 - first_round) * y
        one_hot = tf.one_hot(y, num_classes)

        logits_gravity = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 7])
        # logits = tf.zeros(shape=[FLAGS.batch_size, 1001], dtype=tf.float32)
        #for i in range(2):   ########### 2 is the FLAGS.batch_size
        logits1 = (logits_dict[0][0] * logits_gravity[0][0] + logits_dict[1][0] * logits_gravity[0][1] + logits_dict[2][0] * logits_gravity[0][2] + logits_dict[3][0] * logits_gravity[0][3]
                      + logits_dict[4][0] * logits_gravity[0][4] + logits_dict[5][0] * logits_gravity[0][5] + logits_dict[6][0] * logits_gravity[0][6])/28.0
        logits2 = (logits_dict[0][1] * logits_gravity[1][0] + logits_dict[1][1] * logits_gravity[1][1] + logits_dict[2][1] * logits_gravity[1][2] + logits_dict[3][1] * logits_gravity[1][3]
                   + logits_dict[4][1] * logits_gravity[1][4] + logits_dict[5][1] * logits_gravity[1][5] + logits_dict[6][1] * logits_gravity[1][6]) / 28.0
        logits1 = tf.reshape(logits1, [1, 1001])
        logits2 = tf.reshape(logits2, [1, 1001])
        logits = tf.concat([logits1, logits2], 0)
        # logits = (logits_v3 + 0.25 * logits_adv_v3 + logits_ens3_adv_v3 + \
        #          logits_ens4_adv_v3 + logits_v4 + \
        #          logits_res_v2 + logits_ensadv_res_v2 + logits_resnet) / 7.25
        # auxlogits = (end_points_v3['AuxLogits'] + 0.25 * end_points_adv_v3['AuxLogits'] + end_points_ens3_adv_v3['AuxLogits'] + \
        #             end_points_ens4_adv_v3['AuxLogits'] + end_points_v4['AuxLogits'] + \
        #             end_points_res_v2['AuxLogits'] + end_points_ensadv_res_v2['AuxLogits']) / 6.25
        # cross_entropy = tf.zeros(shape=(FLAGS.batch_size))
        #print(str(gravity))
        cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                        tf.clip_by_value(logits, 1e-8, tf.reduce_max(logits)),
                                                        label_smoothing=0.0,
                                                        weights=1.0)
        # cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
        #                                                  auxlogits,
        #                                                  label_smoothing=0.0,
        #                                                  weights=0.4)

        #print(str(cross_entropy))

        noise = tf.gradients(cross_entropy, x)[0]

        #print(str(tf.gradients(cross_entropy, x)) + '=========')

        # if(noise == None):
        #     noise = tf.zeros(shape=[FLAGS.batch_size, 299, 299, 3])
        #     print('noise出了问题！')
        noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
        grad = momentum * grad + noise

        adv = x + alpha * tf.sign(grad)
        adv = tf.clip_by_value(adv, x_min, x_max)




    # x_adv, y, _, _, _, _, pred = tf.while_loop(stop, graph, [x, y, i, x_max, x_min, grad, pred])


        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        s5 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        s8 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))

        s1.restore(sess, FLAGS.checkpoint_path_inception_v3)
        s2.restore(sess, FLAGS.checkpoint_path_adv_inception_v3)
        s3.restore(sess, FLAGS.checkpoint_path_ens3_adv_inception_v3)
        s4.restore(sess, FLAGS.checkpoint_path_ens4_adv_inception_v3)
        s5.restore(sess, FLAGS.checkpoint_path_inception_v4)
        s6.restore(sess, FLAGS.checkpoint_path_inception_resnet_v2)
        s7.restore(sess, FLAGS.checkpoint_path_ens_adv_inception_resnet_v2)
        s8.restore(sess, FLAGS.checkpoint_path_resnet)


        #sess.run(tf.global_variables_initializer())

        #######
        # test
        # for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        #     logits_v3_,  Pred_dict = sess.run([logits_v3, pred_dict], feed_dict={x: images})
        #     label = np.argmax(Pred_dict, axis=1)
        #     print('label::::::', label)
        #     print('=================\n', logits_v3_)
        #######
        sum = 0
        success_num = 0
        l2_distance = 0
        label_distance = 0
        for filenames, images in load_images(FLAGS.input_dir, batch_shape):
            sum += len(filenames)
            images = images.astype(np.float32)
            images_flatten_initial = images.reshape((2,268203))
            # true_label = []
            # for i in range(len(filenames)):
            #     true_label.append(filenames[i].split('-')[2][:-4])
            #print('image.astype::::::::', images)
            #adv_images = tf.zeros(shape=batch_shape)
            #adv_images = []

            #prediction = 0!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            prediction = []
            for i in range(FLAGS.num_iter):
                logits_proportion = []

                if i == 0:
                    prediction = sess.run(pred, feed_dict={x: images})
                    print('true_label::::::::', prediction)

                    end_points_v3_run, end_points_ens3_adv_v3_run, end_points_ens4_adv_v3_run, end_points_v4_run, end_points_res_v2_run, end_points_ensadv_res_v2_run, end_points_resnet_run = \
                        sess.run([end_points_v3, end_points_ens3_adv_v3, end_points_ens4_adv_v3, end_points_v4, end_points_res_v2, end_points_ensadv_res_v2,
                                  end_points_resnet], feed_dict={x: images, y: prediction})
                else:
                    adv_pred, end_points_v3_run, end_points_ens3_adv_v3_run, end_points_ens4_adv_v3_run, end_points_v4_run, end_points_res_v2_run, end_points_ensadv_res_v2_run, end_points_resnet_run = \
                        sess.run([pred, end_points_v3, end_points_ens3_adv_v3, end_points_ens4_adv_v3, end_points_v4, end_points_res_v2, end_points_ensadv_res_v2,
                                  end_points_resnet], feed_dict={x: images, y: prediction})


                # print('prediction::::::::::', prediction)
                # print('end_points_v3_run[Predictions]:::::::', end_points_v3_run['Predictions'])
                if i == FLAGS.num_iter - 1:
                    print('adv_pred:::::::', adv_pred)

                    for l in range(len(filenames)):
                        label_distance += abs(prediction[l] - adv_pred[l])

                        if int(prediction[l]) == adv_pred[l]:  # The test tag is always 1 more than the real tag, so I subtract 1 here
                            success_num += 1
                            #print('sucess_num:::::', success_num)


                for j in range(FLAGS.batch_size):
                    end_points_v3_Pred = end_points_v3_run['Predictions'][j][prediction[j]]
                    # end_points_adv_v3_Pred = end_points_adv_v3['Predictions']
                    end_points_ens3_adv_v3_Pred = end_points_ens3_adv_v3_run['Predictions'][j][prediction[j]]
                    end_points_ens4_adv_v3_Pred = end_points_ens4_adv_v3_run['Predictions'][j][prediction[j]]
                    end_points_v4_Pred = end_points_v4_run['Predictions'][j][prediction[j]]
                    end_points_res_v2_Pred = end_points_res_v2_run['Predictions'][j][prediction[j]]
                    end_points_ensadv_res_v2_Pred = end_points_ensadv_res_v2_run['Predictions'][j][prediction[j]]
                    end_points_resnet_Pred = end_points_resnet_run['predictions'][j][prediction[j]]

                    ens_Pred_Value = [end_points_v3_Pred, end_points_ens3_adv_v3_Pred,
                                      end_points_ens4_adv_v3_Pred,
                                      end_points_v4_Pred, end_points_res_v2_Pred,
                                      end_points_ensadv_res_v2_Pred,
                                      end_points_resnet_Pred]
                    TopKFitIndx = np.argsort(ens_Pred_Value)
                    # logits_dictionary = tf.placeholder(dtype=tf.int32, shape=(7, FLAGS.batch_size, 1001))
                    # TopKFitIndex = tf.placeholder(dtype=tf.int32, shape=(FLAGS.batch_size))
                    a = [0.0]*7
                    # print('ens_Pred_Value:::::::::', ens_Pred_Value)
                    # print('TopKFitIndx::::::', TopKFitIndx)
                    for m in range(7):
                        a[TopKFitIndx[m]] = 7 - m
                    logits_proportion.append(a)



                # One_hot, Logits_dict, Logits1, Logits2, Logits, images, Cross_Entropy, Noise, gradient = \
                #     sess.run([one_hot, logits_dict, logits1, logits2, logits, adv, cross_entropy, noise, grad],
                #              feed_dict={x: images, logits_gravity: logits_proportion, y: prediction})

                images = sess.run(adv, feed_dict={x: images, logits_gravity: logits_proportion, y: prediction})

                if i == FLAGS.num_iter - 1:
                    images_flatten_adv = images.reshape((2,268203))
                    l2_distance_list = np.linalg.norm((images_flatten_initial - images_flatten_adv), axis=1, keepdims=True)
                    for n in range(len(filenames)):
                        l2_distance += l2_distance_list[n]




                # print('One_hot::::::::::::', One_hot.shape, One_hot)
                # print('Logits_dict::::::::::', Logits_dict)
                # print('Logits1::::::::::',Logits1.shape, Logits1)
                # print('Logits2::::::::::',Logits2.shape, Logits2)
                #
                # print('Logits:::::::::::', Logits)
                # print('Cross_Entropy:::::::::::::', Cross_Entropy)
                # print('Noise:::::::::::', Noise)
                # print('gradient::::::::::', gradient)
            save_images(images, filenames, FLAGS.output_dir)
        # for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        #   #label = sess.run([pred, end_points_v3_Pred, end_points_ens3_adv_v3_Pred, end_points_ens4_adv_v3_Pred, end_points_v4_Pred, end_points_res_v2_Pred, end_points_ensadv_res_v2_Pred, end_points_resnet_Pred],)
        #   adv_images= sess.run([x_adv], feed_dict={x_input: images})
        #   save_images(adv_images, filenames, FLAGS.output_dir)

        print('sum::::', sum)
        print('success_num::::', success_num)
        rate_wrong = success_num / sum
        print('rate_wrong:::::::', rate_wrong)
        print('l2_distance:::::::', l2_distance)
        print('label_distance:::::::', label_distance)
        end = time.clock()
        print('run time:::::', end - start)


if __name__ == '__main__':
    tf.app.run()
