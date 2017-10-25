from __future__ import print_function

import os
from six.moves import urllib
import sys
import tarfile
from datetime import datetime
import argparse
import pickle
from collections import defaultdict

import cv2
import tensorflow as tf

from SneakerNet.code.imageUtils import *
from SneakerNet.code.dataLoading import *
from SneakerNet.code.eval import *
from SneakerNet.code.mobilenet import *

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat


architecture = "mobilenet_v1_1.0_224"
FLAGS = None


def save_graph_to_file(sess, graph, graph_file_name):
  output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
  with gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  return


def create_model_info():
    ''' Grab the model '''

    data_url = "http://download.tensorflow.org/models/"
    data_url += architecture + "_frozen.tgz"
    bottleneck_tensor_name = "MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6:0"
    bottleneck_tensor_size = (14, 14, 512)
    input_width = 224
    input_height = 224
    input_depth = 3
    resized_input_tensor_name = "input:0"
    model_base_name = "frozen_graph.pb"
    model_file_name = os.path.join(architecture, model_base_name)
    input_mean = 127.5
    input_std = 127.5

    return {
      'data_url': data_url,
      'bottleneck_tensor_name': bottleneck_tensor_name,
      'bottleneck_tensor_size': bottleneck_tensor_size,
      'input_width': input_width,
      'input_height': input_height,
      'input_depth': input_depth,
      'resized_input_tensor_name': resized_input_tensor_name,
      'model_file_name': model_file_name,
      'input_mean': input_mean,
      'input_std': input_std,
  }


def maybe_download_and_extract(data_url):
  """Download and extract model tar file.

  If the pretrained model we're using doesn't already exist, this function
  downloads it from the TensorFlow.org website and unpacks it into a directory.

  Args:
    data_url: Web location of the tar file containing the pretrained model.
  """
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = data_url.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename,
                        float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    tf.logging.info('Successfully downloaded', filename, statinfo.st_size,
                    'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def create_model_graph(model_info):
  """"Creates a graph from saved GraphDef file and returns a Graph object.

  Args:
    model_info: Dictionary containing information about the model architecture.

  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
  with tf.Graph().as_default() as graph:
    model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
    with gfile.FastGFile(model_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
          graph_def,
          name='',
          return_elements=[
              model_info['bottleneck_tensor_name'],
              model_info['resized_input_tensor_name'],
          ]))
  return graph, bottleneck_tensor, resized_input_tensor


def add_final_training_ops(class_count, class_count_2, final_tensor_name, bottleneck_tensor,
                           bottleneck_tensor_size, latent_size,
                           prediction_type, loss_ratio):
  """Adds a new softmax and fully-connected layer for training."""

  with tf.name_scope('input'):

    # messing with shapes because the MobileNet layer is of size:
    #    [BatchSize, 7, 7, 1024]
    # SO I'm doing a bunch of reshapes basically
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor,
        shape=[None, bottleneck_tensor_size[0], 
               bottleneck_tensor_size[1],
               bottleneck_tensor_size[2]],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(tf.float32,
                                        [None, class_count],
                                        name='GroundTruthInput')

    if prediction_type == "both":
        ground_truth_input_2 = tf.placeholder(tf.float32,
                                              [None, class_count_2],
                                              name='GroundTruthInput2')
    else:
        ground_truth_input_2 = None

  if prediction_type != "both":

    with tf.name_scope("mobilenet_retrain"):
        conv_intermediate = mobilenet_conv_block(
            bottleneck_input, "intermediate_conv", stride=2
        )
        squeezed = mobilenet_top_block(conv_intermediate, latent_size,
                                       "mobilenet_top")

  else:

    with tf.name_scope("mobilenet_retrain"):
      conv_intermediate = mobilenet_conv_block(
          bottleneck_input, "intermediate_conv", stride=2
      )

      with tf.name_scope("conv_top_block"):
          squeezed1 = mobilenet_top_block(conv_intermediate, 
                                          latent_size,
                                          "mobilenet_top_sneaker")

          squeezed2 = mobilenet_top_block(conv_intermediate, 
                                          latent_size,
                                          "mobilenet_top_brand")

  
  if prediction_type != "both":

    with tf.name_scope('weights'):
      initial_value = tf.truncated_normal(
          [latent_size, class_count], stddev=0.001)
      layer_weights = tf.Variable(initial_value, name='final_weights')

    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
      # variable_summaries(layer_biases)

    with tf.name_scope('Wx_plus_b'):
      logits = tf.matmul(squeezed, layer_weights) + layer_biases
      tf.summary.histogram('pre_activations', logits)

    final_tensor_1 = tf.nn.softmax(logits, name=final_tensor_name)
    tf.summary.histogram('activations', final_tensor_1)

    with tf.name_scope('cross_entropy'):
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
          labels=ground_truth_input, logits=logits)
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
      optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
      train_step = optimizer.minimize(cross_entropy_mean)

    # don't need two prediction tensors
    final_tensor_2 = None

  else:
      
    # concatenate the two mobilenet top blocks
    squeezed = tf.concat([squeezed1, squeezed2], 1)

    with tf.name_scope('fullyconnected'):

      with tf.name_scope('biases'):
          sneaker_biases = tf.Variable(tf.zeros([class_count]),
                                       name='sneaker_biases')
          brand_biases = tf.Variable(tf.zeros([class_count_2]),
                                     name='brand_biases')

      with tf.name_scope('weights'):
          sneaker_weights = tf.Variable(tf.truncated_normal(
              [latent_size * 2, class_count], stddev=0.001),
              name="sneaker_weights")
          brand_weights = tf.Variable(tf.truncated_normal(
              [latent_size * 2, class_count_2], stddev=0.001),
              name="brand_weights")

      with tf.name_scope('fc_final'):
          logits1 = tf.matmul(squeezed, sneaker_weights) + sneaker_biases
          logits2 = tf.matmul(squeezed, brand_weights) + brand_biases

          # use brand logits to inform the sneaker logits
          # combine_biases = tf.Variable(tf.zeros([class_count]),
          #                              name='combine_biases')
          # combine_weights = tf.Variable(tf.truncated_normal(
          #     [class_count_2, class_count], stddev=0.001),
          #     name='combine_weights')

          # just straight adding for now, weights will learn the mapping
          # logits1 = tf.add(
          #     tf.matmul(logits2, combine_weights) + combine_biases,
          #     logits1
          # )

          final_tensor_1 = tf.nn.softmax(logits1, name="Final_Sneakers")
          final_tensor_2 = tf.nn.softmax(logits2, name="Final_Brands")

      with tf.name_scope('cross_entropy'):
          cross_entropy_1 = tf.nn.softmax_cross_entropy_with_logits(
              labels=ground_truth_input, logits=logits1)
          cross_entropy_2 = tf.nn.softmax_cross_entropy_with_logits(
              labels=ground_truth_input_2, logits=logits2)

          # weigh one loss more than the other
          sneaker_ratio = tf.constant(loss_ratio)
          brand_ratio = tf.constant(1 - loss_ratio)

          # use a weighted loss!
          cross_entropy_mean = \
              tf.add(
                  tf.multiply(sneaker_ratio, tf.reduce_mean(cross_entropy_1)), \
                  tf.multiply(brand_ratio, tf.reduce_mean(cross_entropy_2))
              )

          tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):

        optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input,
          ground_truth_input, ground_truth_input_2, final_tensor_1,
          final_tensor_2)


def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)

    print("Setting model info...")
    model_info = create_model_info()
    maybe_download_and_extract(model_info["data_url"])

    print("Setting up graph...")
    graph, bottleneck_tensor, resized_image_tensor = \
        create_model_graph(model_info)

    print("Generating image list...")
    # save if necessary
    # loading it takes a long time so it's nice to be able to save it
    if FLAGS.save_image_split:
        print("Loading new image split file...")
        image_lists = create_image_lists(FLAGS.image_dir,
                                         FLAGS.validation_percentage,
                                         prediction_type=FLAGS.prediction_type)
        split_file = open(FLAGS.image_split_file, "w")
        pickle.dump(image_lists, split_file)
        split_file.close()
    else:
        print("Loading image split file...")
        image_lists = pickle.load(open(FLAGS.image_split_file, 'r'))

    image_labels = list(image_lists.keys())
    class_count = len(image_labels)
    class_count_2 = len(set([s.split("--")[0] for s in image_labels]))

    print("Beginning session...")
    with tf.Session(graph=graph) as sess:

        # add new image decoding ops to the graph
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
            model_info['input_width'], model_info['input_height'],
            model_info['input_depth'], model_info['input_mean'],
            model_info['input_std'])

        # add distortion ops to the graph
        distorted_jpeg_data_tensor, distorted_image_tensor = \
            add_input_distortions(
                FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
                FLAGS.random_brightness, model_info['input_width'],
                model_info['input_height'], model_info['input_depth'],
                model_info['input_mean'], model_info['input_std'])

        # add the new top of the graph
        (train_step, cross_entropy, bottleneck_input, ground_truth_input,
         ground_truth_input_2, final_tensor_1, final_tensor_2) = add_final_training_ops(
             class_count, class_count_2, FLAGS.final_tensor_name, bottleneck_tensor,
             model_info["bottleneck_tensor_size"], FLAGS.latent_size,
             FLAGS.prediction_type, FLAGS.loss_ratio)

        if FLAGS.prediction_type != "both":

            # set up evaluation metrics
            with tf.name_scope('accuracy'):

                with tf.name_scope('correct_prediction'):
                    prediction = tf.argmax(final_tensor_1, 1)
                    correct_prediction = \
                        tf.equal(prediction, tf.argmax(ground_truth_input, 1))

                with tf.name_scope('accuracy'):
                    evaluation_step = tf.reduce_mean(
                        tf.cast(correct_prediction, tf.float32))

            with tf.name_scope('top_k_accuracy'):

                labels = tf.cast(tf.argmax(ground_truth_input, axis=1),
                                 tf.int32)

                eval_top_5 = tf.reduce_mean(tf.cast(
                    tf.nn.in_top_k(predictions=final_tensor_1,
                                   targets=labels,
                                   k=5), tf.float32))


                eval_top_3 = tf.reduce_mean(tf.cast(
                    tf.nn.in_top_k(predictions=final_tensor_1,
                                   targets=labels,
                                   k=3), tf.float32))


            tf.summary.scalar('accuracy', evaluation_step)
            tf.summary.scalar('top_5', eval_top_5)
            tf.summary.scalar('top_3', eval_top_3)

        else:

            # we're predicting two separate sets of classes, so lots of eval

            with tf.name_scope('accuracy'):

                with tf.name_scope('correct_prediction'):
                    prediction_1 = tf.argmax(final_tensor_1, 1)
                    prediction_2 = tf.argmax(final_tensor_2, 1)
                    correct_prediction_1 = \
                        tf.equal(prediction_1, tf.argmax(ground_truth_input, 1))
                    correct_prediction_2 = \
                        tf.equal(prediction_2, tf.argmax(ground_truth_input_2, 1))

                with tf.name_scope('accuracy'):
                    evaluation_sneaker = tf.reduce_mean(
                        tf.cast(correct_prediction_1, tf.float32))
                    evaluation_brand = tf.reduce_mean(
                        tf.cast(correct_prediction_2, tf.float32))

            with tf.name_scope('top_k_accuracy'):

                labels_1 = tf.cast(tf.argmax(ground_truth_input, axis=1),
                                   tf.int32)
                labels_2 = tf.cast(tf.argmax(ground_truth_input_2, axis=1),
                                   tf.int32)

                eval_top_5 = tf.reduce_mean(tf.cast(
                    tf.nn.in_top_k(predictions=final_tensor_1,
                                   targets=labels_1,
                                   k=5), tf.float32))

                eval_top_5_brand = tf.reduce_mean(tf.cast(
                    tf.nn.in_top_k(predictions=final_tensor_2,
                                   targets=labels_2,
                                   k=5), tf.float32))

                eval_top_3 = tf.reduce_mean(tf.cast(
                    tf.nn.in_top_k(predictions=final_tensor_1,
                                   targets=labels_1,
                                   k=3), tf.float32))

                eval_top_3_brand = tf.reduce_mean(tf.cast(
                    tf.nn.in_top_k(predictions=final_tensor_2,
                                   targets=labels_2,
                                   k=3), tf.float32))

            tf.summary.scalar('sneaker_accuracy', evaluation_sneaker)
            tf.summary.scalar('brand_accuracy', evaluation_brand)
            tf.summary.scalar('sneaker_top_5', eval_top_5)
            tf.summary.scalar('sneaker_top_3', eval_top_3)
            tf.summary.scalar('brand_top_5', eval_top_5_brand)
            tf.summary.scalar('brand_top_3', eval_top_3_brand)

        merged = tf.summary.merge_all()

        train_log_dir = os.path.join(
            FLAGS.summaries_dir, "train/{}".format(FLAGS.run_name))
        if not os.path.isdir(train_log_dir):
            os.makedirs(train_log_dir)
        train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

        val_log_dir = os.path.join(
            FLAGS.summaries_dir, "validation/{}".format(FLAGS.run_name))
        if not os.path.isdir(val_log_dir):
            os.makedirs(val_log_dir)
        validation_writer = tf.summary.FileWriter(val_log_dir)

        init = tf.global_variables_initializer()
        sess.run(init)


        sneaker_labels = sorted(list(image_lists.keys()))
        brand_labels = sorted(
            list(set([s.split("--")[0] for s in sneaker_labels]))
        )


        # RUN IT ALL
        for i in range(FLAGS.how_many_training_steps):

            # since we're always doing distortion, calculate bottlenecks
            # on the fly
            (train_bottlenecks, train_ground_truth, train_ground_truth_2) = \
                get_random_distorted_bottlenecks(
                    sess, image_lists, FLAGS.train_batch_size, 'training',
                    FLAGS.image_dir, distorted_jpeg_data_tensor,
                    distorted_image_tensor, resized_image_tensor,
                    bottleneck_tensor, FLAGS.prediction_type)

            if FLAGS.prediction_type != "both":
                train_summary, _ = sess.run(
                    [merged, train_step],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth})
            else:
                train_summary, _ = sess.run(
                    [merged, train_step],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth,
                               ground_truth_input_2: train_ground_truth_2})

            train_writer.add_summary(train_summary, i)

            # print out some results occasionally
            if (i % FLAGS.train_eval_step_interval) == 0:

                if FLAGS.prediction_type != "both":
                    train_accuracy, cross_entropy_value = sess.run(
                        [evaluation_step, cross_entropy],
                        feed_dict={bottleneck_input: train_bottlenecks,
                                   ground_truth_input: train_ground_truth})

                    # log training results
                    tf.logging.info("{}: Step {}: Train Acc = {:.1f}".format(
                        datetime.now(), i, train_accuracy * 100))
                    tf.logging.info("{}: Step {}: Train Loss = {}".format(
                        datetime.now(), i, cross_entropy_value))

                else:
                    train_accuracy_sneaker, train_accuracy_brand, \
                        cross_entropy_value = sess.run(
                            [evaluation_sneaker, evaluation_brand,
                             cross_entropy],
                            feed_dict={bottleneck_input: train_bottlenecks,
                                       ground_truth_input: train_ground_truth,
                                       ground_truth_input_2: train_ground_truth_2})

                    # log training results
                    tf.logging.info("{}: Step {}: Train Acc (SNKR) = {:.1f}".format(
                        datetime.now(), i, train_accuracy_sneaker * 100))
                    tf.logging.info("{}: Step {}: Train Acc (BRND) = {:.1f}".format(
                        datetime.now(), i, train_accuracy_brand * 100))
                    tf.logging.info("{}: Step {}: Train Loss = {}".format(
                        datetime.now(), i, cross_entropy_value))

            if (i % FLAGS.val_eval_step_interval) == 0:

                # don't do distortions for validation data
                valid_bottlenecks, valid_ground_truth, valid_ground_truth_2, _ = \
                    get_random_cached_bottlenecks(
                        sess, image_lists, FLAGS.validation_batch_size,
                        'validation', FLAGS.bottleneck_dir, FLAGS.image_dir,
                        jpeg_data_tensor, decoded_image_tensor,
                        resized_image_tensor, bottleneck_tensor,
                        FLAGS.architecture,
                        prediction_type=FLAGS.prediction_type)

                if FLAGS.prediction_type != "both":

                    validation_summary, validation_accuracy, val_top_5, val_top_3 = \
                            sess.run(
                                [merged, evaluation_step, eval_top_5, eval_top_3],
                                feed_dict={bottleneck_input: valid_bottlenecks,
                                           ground_truth_input: valid_ground_truth})

                    validation_writer.add_summary(validation_summary, i)

                    log_string = "{}: Step {}: Val Acc = {:.1f}," \
                            + " Val Top 3 = {:.1f}, Val Top 5 = {:.1f}"
                    tf.logging.info(log_string.format(
                        datetime.now(), i, validation_accuracy * 100,
                        val_top_3 * 100, val_top_5 * 100))

                else:

                    valid_summary, valid_accuracy_sneaker, val_top_5, val_top_3, \
                        valid_accuracy_brand, val_top_5_brand, val_top_3_brand = \
                            sess.run(
                                [merged, evaluation_sneaker, eval_top_5,
                                 eval_top_3, evaluation_brand,
                                 eval_top_5_brand, eval_top_3_brand],
                                feed_dict={bottleneck_input: valid_bottlenecks,
                                           ground_truth_input:
                                           valid_ground_truth,
                                           ground_truth_input_2:
                                           valid_ground_truth_2})

                    validation_writer.add_summary(valid_summary, i)

                    log_string = "{}: Step {}: Val Acc (SNKR) = {:.1f}," \
                            + " Val Top 3 = {:.1f}, Val Top 5 = {:.1f}"
                    tf.logging.info(log_string.format(
                        datetime.now(), i, valid_accuracy_sneaker * 100,
                        val_top_3 * 100, val_top_5 * 100))
                    log_string_2 = "{}: Step {}: Val Acc (BRND) = {:.1f}," \
                            + " Val Top 3 = {:.1f}, Val Top 5 = {:.1f}"
                    tf.logging.info(log_string_2.format(
                        datetime.now(), i, valid_accuracy_brand * 100,
                        val_top_3_brand * 100, val_top_5_brand * 100))



            ###
            # Here's where we'll put the intermediate graph saving code
            ###

        # run inference on the entire validation set once training is completed
        valid_bottlenecks, valid_ground_truth, valid_ground_truth_2, fnames = \
            get_random_cached_bottlenecks(
                sess, image_lists, -1, 'validation', FLAGS.bottleneck_dir,
                FLAGS.image_dir, jpeg_data_tensor, decoded_image_tensor,
                resized_image_tensor, bottleneck_tensor, FLAGS.architecture,
                prediction_type=FLAGS.prediction_type)

        final_valid = defaultdict(int)
        class_accs_1 = [0] * len(valid_ground_truth[0])
        class_accs_2 = [0] * len(valid_ground_truth_2[0])
        class_1_n = [0] * len(valid_ground_truth[0])
        class_2_n = [0] * len(valid_ground_truth_2[0])

        n_correct = 0

        sneaker_labels = sorted(list(image_lists.keys()))
        brand_labels = sorted(
            list(set([s.split("--")[0] for s in sneaker_labels]))
        )

        # feed in one by one because entire validation set can't fit in memory
        for ii in range(len(valid_bottlenecks)):

            valid_accuracy_sneaker, val_top_5, val_top_3, \
                valid_accuracy_brand, val_top_5_brand, val_top_3_brand, \
                pred_1, pred_2 = \
                    sess.run(
                        [evaluation_sneaker, eval_top_5, eval_top_3,
                         evaluation_brand, eval_top_5_brand, eval_top_3_brand,
                         prediction_1, prediction_2],
                        feed_dict={bottleneck_input: [valid_bottlenecks[ii]],
                                   ground_truth_input:
                                   [valid_ground_truth[ii]],
                                   ground_truth_input_2:
                                   [valid_ground_truth_2[ii]]})

            if valid_accuracy_sneaker > 0:
                n_correct += 1

            final_valid["sneaker acc"] += valid_accuracy_sneaker
            final_valid["sneaker top 5"] += val_top_5
            final_valid["sneaker top 3"] += val_top_3
            final_valid["brand acc"] += valid_accuracy_brand
            final_valid["brand top 5"] += val_top_5_brand
            final_valid["brand top 3"] += val_top_3_brand

            if ii % 100 == 0:
                print("{} / {} validations".format(ii, len(valid_bottlenecks)))

            # check class-by-class accuracy
            class_1 = np.argmax(valid_ground_truth[ii])
            class_2 = np.argmax(valid_ground_truth_2[ii])
            class_accs_1[class_1] += valid_accuracy_sneaker
            class_accs_2[class_2] += valid_accuracy_brand
            class_1_n[class_1] += 1
            class_2_n[class_2] += 1

            print()
            print("Loaded from: {}".format(fnames[ii]))
            print(
                "Correct sneaker class: {}".format(sneaker_labels[class_1])
            )
            print(
                "Correct brand class: {}".format(brand_labels[class_2])
            )
            print(
                "Predicted sneaker class: {}".format(sneaker_labels[pred_1])
            )
            print(
                "Predicted brand class: {}".format(brand_labels[pred_2])
            )

        print("Correct sneaker predictions: {}".format(n_correct))

        # normalize all values
        for key in final_valid.keys():
            final_valid[key] /= float(len(valid_bottlenecks))

        for i, v in enumerate(class_accs_1):
            try:
                class_accs_1[i] /= float(class_1_n[i])
            except:
                continue

        for i, v in enumerate(class_accs_2):
            try:
                class_accs_2[i] /= float(class_2_n[i])
            except:
                continue

        # print results:

        print()
        print("---------------------------")
        print("FINAL VALIDATION RESULTS:")
        for key in final_valid.keys():
            print("{}: {}".format(key, 100.0 * final_valid[key]))
        print()
        for i, v in enumerate(class_accs_1):
            print("{}: {}".format(sneaker_labels[i], 100.0 * v))
        print()
        for i, v in enumerate(class_accs_2):
            print("{}: {}".format(brand_labels[i], 100.0 * v))

        # save final trained graph and the order for the labels
        save_graph_to_file(sess, graph, FLAGS.output_graph)
        with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')


if __name__ == "__main__":
    
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_split_file',
      type=str,
      default='../data/image_split_paths.p',
      help='Path to file for pickling the training/val split'
  )
  parser.add_argument(
      '--save_image_split',
      type=int,
      default=0,
      help='Boolean identifying whether or not to save the split'
  )
  parser.add_argument(
      '--image_dir',
      type=str,
      default='../data/images',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--output_graph',
      type=str,
      default='../models/trained/output_graph.pb',
      help='Where to save the trained graph.'
  )
  parser.add_argument(
      '--intermediate_output_graphs_dir',
      type=str,
      default='../models/intermediate/intermediate_graph/',
      help='Where to save the intermediate graphs.'
  )
  parser.add_argument(
      '--intermediate_store_frequency',
      type=int,
      default=400,
      help="""\
         How many steps to store intermediate graph. If "0" then will not
         store.\
      """
  )
  parser.add_argument(
      '--output_labels',
      type=str,
      default='../models/output_labels.txt',
      help='Where to save the trained graph\'s labels.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='../logs/retrain_logs',
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=5000,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,
      help='How large a learning rate to use when training.'
  )
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=0.1,
      help='What proportion of images to use as a test set.'
  )
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=0.1,
      help='What proportion of images to use as a validation set.'
  )
  parser.add_argument(
      '--train_eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--val_eval_step_interval',
      type=int,
      default=20,
      help='Hpw often to evaluate the validation results.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=500,
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
  )
  parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
  )
  parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--model_dir',
      type=str,
      default='../models/pretrained/',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='../data/bottleneck',
      help='Path to cache bottleneck layer values as files.'
  )
  parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result_1',
      help="""\
      The name of the output classification layer in the retrained graph.\
      """
  )
  parser.add_argument(
      '--flip_left_right',
      default=True,
      help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--random_crop',
      type=int,
      default=20,
      help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
  )
  parser.add_argument(
      '--random_scale',
      type=int,
      default=20,
      help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
  )
  parser.add_argument(
      '--random_brightness',
      type=int,
      default=20,
      help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
  )
  parser.add_argument(
      '--architecture',
      type=str,
      default='mobilenet_1.0_224',
      help="""\
      Which model architecture to use. 'inception_v3' is the most accurate, but
      also the slowest. For faster or smaller models, chose a MobileNet with the
      form 'mobilenet_<parameter size>_<input_size>[_quantized]'. For example,
      'mobilenet_1.0_224' will pick a model that is 17 MB in size and takes 224
      pixel input images, while 'mobilenet_0.25_128_quantized' will choose a much
      less accurate, but smaller and faster network that's 920 KB on disk and
      takes 128x128 images. See https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
      for more information on Mobilenet.\
      """)
  parser.add_argument(
      '--run_name',
      type=str,
      default="default",
      help="""\
      Name of directory to save logs in for TensorBoard
      """)
  parser.add_argument(
      '--latent_size',
      type=int,
      default=1000,
      help="""\
      Size of second-last FC layer
      """)
  parser.add_argument(
      '--prediction_type',
      type=str,
      default="brands",
      help="""\
      Type of classes to predict. For predicting brands, put brands.
      For predicting sneakers, put sneakers.
      """)
  parser.add_argument(
      '--loss_ratio',
      type=float,
      default=0.5,
      help="""\
      Proportion of sneaker loss to brand loss in combined loss value
      """)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
