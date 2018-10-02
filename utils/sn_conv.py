import tensorflow as tf

slim = tf.contrib.slim
from tensorflow.contrib.framework.python.ops import add_arg_scope

from utils.slim_utils import _build_variable_getter, _add_variable_to_collections, convert_data_format
from tensorflow.python.ops import nn_ops


@add_arg_scope
def convolution(inputs,
                num_outputs,
                kernel_size=[3, 3],
                spectral_normalization=False,
                stride=1,
                padding='SAME',
                data_format=None,
                rate=1,
                activation_fn=tf.nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=tf.zeros_initializer(),
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None):
  """Adds an N-D convolution followed by an optional batch_norm layer.

  It is required that 1 <= N <= 3.

  `convolution` creates a variable called `weights`, representing the
  convolutional kernel, that is convolved (actually cross-correlated) with the
  `inputs` to produce a `Tensor` of activations. If a `normalizer_fn` is
  provided (such as `batch_norm`), it is then applied. Otherwise, if
  `normalizer_fn` is None and a `biases_initializer` is provided then a `biases`
  variable would be created and added the activations. Finally, if
  `activation_fn` is not `None`, it is applied to the activations as well.

  Performs atrous convolution with input stride/dilation rate equal to `rate`
  if a value > 1 for any dimension of `rate` is specified.  In this case
  `stride` values != 1 are not supported.

  Args:
    inputs: A Tensor of rank N+2 of shape
      `[batch_size] + input_spatial_shape + [in_channels]` if data_format does
      not start with "NC" (default), or
      `[batch_size, in_channels] + input_spatial_shape` if data_format starts
      with "NC".
    num_outputs: Integer, the number of output filters.
    kernel_size: A sequence of N positive integers specifying the spatial
      dimensions of the filters.  Can be a single integer to specify the same
      value for all spatial dimensions.
    stride: A sequence of N positive integers specifying the stride at which to
      compute output.  Can be a single integer to specify the same value for all
      spatial dimensions.  Specifying any `stride` value != 1 is incompatible
      with specifying any `rate` value != 1.
    padding: One of `"VALID"` or `"SAME"`.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".
      For N=3, the valid values are "NDHWC" (default) and "NCDHW".
    rate: A sequence of N positive integers specifying the dilation rate to use
      for atrous convolution.  Can be a single integer to specify the same
      value for all spatial dimensions.  Specifying any `rate` value != 1 is
      incompatible with specifying any `stride` value != 1.
    activation_fn: Activation function. The default value is a ReLU function.
      Explicitly set it to None to skip it and maintain a linear activation.
    normalizer_fn: Normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: Normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for all the variables or
      a dictionary containing a different list of collection per variable.
    outputs_collections: Collection to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_scope`.

  Returns:
    A tensor representing the output of the operation.

  Raises:
    ValueError: If `data_format` is invalid.
    ValueError: Both 'rate' and `stride` are not uniformly 1.
  """
  if data_format not in [None, 'NWC', 'NCW', 'NHWC', 'NCHW', 'NDHWC', 'NCDHW']:
    raise ValueError('Invalid data_format: %r' % (data_format,))

  layer_variable_getter = _build_variable_getter(
      {'bias': 'biases', 'kernel': 'weights'})

  with tf.variable_scope(
      scope, 'Conv', [inputs], reuse=reuse,
      custom_getter=layer_variable_getter) as sc:
    inputs = tf.convert_to_tensor(inputs)
    input_rank = inputs.get_shape().ndims

    if input_rank != 4:
        raise ValueError('Invalid input rank: %i' % (input_rank,))

    df = ('channels_first' if data_format and data_format.startswith('NC')
          else 'channels_last')
    layer = SpectralNormalizedConvolution2D(
        filters=num_outputs,
        kernel_size=kernel_size,
        spectral_normalization=spectral_normalization,
        strides=stride,
        padding=padding,
        data_format=df,
        dilation_rate=rate,
        activation=None,
        use_bias=not normalizer_fn and biases_initializer,
        kernel_initializer=weights_initializer,
        bias_initializer=biases_initializer,
        kernel_regularizer=weights_regularizer,
        bias_regularizer=biases_regularizer,
        activity_regularizer=None,
        trainable=trainable,
        name=sc.name,
        dtype=inputs.dtype.base_dtype,
        _scope=sc,
        _reuse=reuse)
    outputs = layer.apply(inputs)

    # Add variables to collections.
    _add_variable_to_collections(layer.kernel, variables_collections, 'weights')
    if layer.use_bias:
      _add_variable_to_collections(layer.bias, variables_collections, 'biases')

    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return slim.utils.collect_named_outputs(outputs_collections, sc.name, outputs)


NO_OPS = 'NO_OPS'


def spectral_normed_weight(W, u=None, num_iters=1, update_collection=None, with_sigma=False):
  # Usually num_iters = 1 will be enough
  W_shape = W.shape.as_list()
  W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
  if u is None:
    u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

  def power_iteration(i, u_i, v_i):
    v_ip1 = tf.nn.l2_normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
    u_ip1 = tf.nn.l2_normalize(tf.matmul(v_ip1, W_reshaped))
    return i + 1, u_ip1, v_ip1

  _, u_final, v_final = tf.while_loop(
    cond=lambda i, _1, _2: i < num_iters,
    body=power_iteration,
    loop_vars=(tf.constant(0, dtype=tf.int32),
               u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
  )

  if update_collection is None:
    # warnings.warn('Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
    #               '. Please consider using a update collection instead.')
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
    W_bar = W_reshaped / sigma
    with tf.control_dependencies([u.assign(u_final)]):
      W_bar = tf.reshape(W_bar, W_shape)
  else:
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
    W_bar = W_reshaped / sigma
    W_bar = tf.reshape(W_bar, W_shape)
    # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
    # has already been collected on the first call.
    if update_collection != NO_OPS:
      tf.add_to_collection(update_collection, u.assign(u_final))

  if with_sigma:
    return W_bar, sigma
  else:
    return W_bar


class SpectralNormalizedConvolution2D(tf.layers.Conv2D):
  def __init__(self, filters,
               kernel_size,
               spectral_normalization=False,
               strides=(1, 1),
               padding='valid',
               data_format='channels_last',
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(SpectralNormalizedConvolution2D, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name, **kwargs)
    self.spectral_normalization = spectral_normalization

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis].value
    kernel_shape = self.kernel_size + (input_dim, self.filters)

    self.kernel = self.add_variable(name='kernel',
                                    shape=kernel_shape,
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint,
                                    trainable=True,
                                    dtype=self.dtype)
    if self.spectral_normalization:
      # TODO pass update_collection?
      vs = tf.get_variable_scope()
      upd_coll = None if not vs.reuse else NO_OPS
      # print("update collection = ", upd_coll)
      self.kernel = spectral_normed_weight(self.kernel, update_collection=upd_coll)
    if self.use_bias:
      self.bias = self.add_variable(name='bias',
                                    shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    trainable=True,
                                    dtype=self.dtype)
    else:
      self.bias = None
    self.input_spec = tf.layers.InputSpec(ndim=self.rank + 2,
                                          axes={channel_axis: input_dim})
    self._convolution_op = nn_ops.Convolution(
        input_shape,
        filter_shape=self.kernel.get_shape(),
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=self.padding.upper(),
        data_format=convert_data_format(self.data_format,
                                        self.rank + 2))
    self.built = True


class SpectralNormalizedConvolution2DTranspose(tf.layers.Conv2DTranspose):
  def __init__(self, filters,
               kernel_size,
               spectral_normalization=False,
               strides=(1, 1),
               padding='valid',
               data_format='channels_last',
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(SpectralNormalizedConvolution2DTranspose, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name, **kwargs)
    self.spectral_normalization = spectral_normalization

  def build(self, input_shape):
    if len(input_shape) != 4:
      raise ValueError('Inputs should have rank 4. Received input shape: ' +
                       str(input_shape))
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis]
    self.input_spec = tf.layers.InputSpec(ndim=4, axes={channel_axis: input_dim})
    kernel_shape = self.kernel_size + (self.filters, input_dim)

    self.kernel = self.add_variable(name='kernel',
                                    shape=kernel_shape,
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint,
                                    trainable=True,
                                    dtype=self.dtype)

    if self.spectral_normalization:
      # TODO pass update_collection?
      vs = tf.get_variable_scope()
      upd_coll = None if not vs.reuse else NO_OPS
      self.kernel = spectral_normed_weight(self.kernel, update_collection=upd_coll)

    if self.use_bias:
      self.bias = self.add_variable(name='bias',
                                    shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    trainable=True,
                                    dtype=self.dtype)
    else:
      self.bias = None
    self.built = True



@add_arg_scope
def convolution2d_transpose(
    inputs,
    num_outputs,
    kernel_size,
    stride=1,
    spectral_normalization=False,
    padding='SAME',
    data_format='NHWC',
    activation_fn=tf.nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=tf.contrib.layers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(),
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None):
  """Adds a convolution2d_transpose with an optional batch normalization layer.

  The function creates a variable called `weights`, representing the
  kernel, that is convolved with the input. If `normalizer_fn` is `None`, a
  second variable called 'biases' is added to the result of the operation.

  Args:
    inputs: A 4-D `Tensor` of type `float` and shape
      `[batch, height, width, in_channels]` for `NHWC` data format or
      `[batch, in_channels, height, width]` for `NCHW` data format.
    num_outputs: Integer, the number of output filters.
    kernel_size: A list of length 2 holding the [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    stride: A list of length 2: [stride_height, stride_width].
      Can be an int if both strides are the same.  Note that presently
      both strides must have the same value.
    padding: One of 'VALID' or 'SAME'.
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    activation_fn: Activation function. The default value is a ReLU function.
      Explicitly set it to None to skip it and maintain a linear activation.
    normalizer_fn: Normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: Normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for all the variables or
      a dictionary containing a different list of collection per variable.
    outputs_collections: Collection to add the outputs.
    trainable: Whether or not the variables should be trainable or not.
    scope: Optional scope for variable_scope.

  Returns:
    A tensor representing the output of the operation.

  Raises:
    ValueError: If 'kernel_size' is not a list of length 2.
    ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: If `C` dimension of `inputs` is None.
  """
  layer_variable_getter = _build_variable_getter(
      {'bias': 'biases', 'kernel': 'weights'})

  with tf.variable_scope(
      scope, 'Conv2d_transpose', [inputs], reuse=reuse,
      custom_getter=layer_variable_getter) as sc:
    if data_format not in ('NCHW', 'NHWC'):
      raise ValueError('data_format has to be either NCHW or NHWC.')

    inputs = tf.convert_to_tensor(inputs)

    df = ('channels_first' if data_format and data_format.startswith('NC')
          else 'channels_last')
    layer = SpectralNormalizedConvolution2DTranspose(
        filters=num_outputs,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        spectral_normalization=False,
        data_format=df,
        activation=None,
        use_bias=not normalizer_fn and biases_initializer,
        kernel_initializer=weights_initializer,
        bias_initializer=biases_initializer,
        kernel_regularizer=weights_regularizer,
        bias_regularizer=biases_regularizer,
        activity_regularizer=None,
        trainable=trainable,
        name=sc.name,
        dtype=inputs.dtype.base_dtype,
        _scope=sc,
        _reuse=reuse)
    outputs = layer.apply(inputs)

    # Add variables to collections.
    _add_variable_to_collections(layer.kernel, variables_collections, 'weights')
    if layer.bias is not None:
      _add_variable_to_collections(layer.bias, variables_collections, 'biases')

    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return slim.utils.collect_named_outputs(outputs_collections, sc.name, outputs)
