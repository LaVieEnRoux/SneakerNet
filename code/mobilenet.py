from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim


def mobilenet_conv_block(input_tensor, scope_prefix,
                         kernel=[3, 3], stride=1,
                         depth=1024):

    with tf.name_scope(scope_prefix):
      net = slim.separable_conv2d(input_tensor, None, kernel,
                                  depth_multiplier=1,
                                  stride=stride,
                                  rate=1,
                                  normalizer_fn=slim.batch_norm)

      net = slim.conv2d(net, depth, [1, 1],
                        stride=1,
                        normalizer_fn=slim.batch_norm)

      return net


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.
  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are large enough.
  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]
  Returns:
    a tensor with the kernel size.
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out


def mobilenet_top_block(input_tensor, latent_size, scope_prefix):

  convOut = mobilenet_conv_block(input_tensor, scope_prefix + "_final_conv_block")

  layer_name = scope_prefix + '_output'
  with tf.name_scope(layer_name):

      kernel_size = _reduced_kernel_size_for_small_input(convOut, [7, 7])
      pooled = tf.contrib.slim.avg_pool2d(convOut, kernel_size,
                                          padding='VALID')

      dropout = tf.contrib.slim.dropout(pooled, keep_prob=0.5)

      conv1d = tf.contrib.slim.conv2d(dropout, latent_size, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None)

      squeezed = tf.squeeze(conv1d, [1, 2], name="SpatialSqueeze")

  return squeezed
