"""
The whole purpose of this file is to make slim hacking a bit easier
TF removes all intermediate functions from namespace
so just copy pasting a few slim functions to rewrite them does not really work
as it relies on internal functions unavailable from outside.

So everything below is copy pasted from TF sources. Honestly I believe I should not have
to do it, but oh wey...
"""
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils

slim = tf.contrib.slim


def _model_variable_getter(getter, name, shape=None, dtype=None,
                           initializer=None, regularizer=None, trainable=True,
                           collections=None, caching_device=None,
                           partitioner=None, rename=None, use_resource=None,
                           **_):
  """Getter that uses model_variable for compatibility with core layers."""
  short_name = name.split('/')[-1]
  if rename and short_name in rename:
    name_components = name.split('/')
    name_components[-1] = rename[short_name]
    name = '/'.join(name_components)
  return slim.model_variable(
      name, shape=shape, dtype=dtype, initializer=initializer,
      regularizer=regularizer, collections=collections, trainable=trainable,
      caching_device=caching_device, partitioner=partitioner,
      custom_getter=getter, use_resource=use_resource)


def _build_variable_getter(rename=None):
  """Build a model variable getter that respects scope getter and renames."""
  # VariableScope will nest the getters
  def layer_variable_getter(getter, *args, **kwargs):
    kwargs['rename'] = rename
    return _model_variable_getter(getter, *args, **kwargs)
  return layer_variable_getter


def _add_variable_to_collections(variable, collections_set, collections_name):
  """Adds variable (or all its parts) to all collections with that name."""
  collections = utils.get_variable_collections(
      collections_set, collections_name) or []
  variables_list = [variable]
  # if isinstance(variable, tf.python.ops.variables.PartitionedVariable):
  #   variables_list = [v for v in variable]
  for collection in collections:
    for var in variables_list:
      if var not in tf.get_collection(collection):
        tf.add_to_collection(collection, var)


def convert_data_format(data_format, ndim):
  if data_format == 'channels_last':
    if ndim == 3:
      return 'NWC'
    elif ndim == 4:
      return 'NHWC'
    elif ndim == 5:
      return 'NDHWC'
    else:
      raise ValueError('Input rank not supported:', ndim)
  elif data_format == 'channels_first':
    if ndim == 3:
      return 'NCW'
    elif ndim == 4:
      return 'NCHW'
    elif ndim == 5:
      return 'NCDHW'
    else:
      raise ValueError('Input rank not supported:', ndim)
  else:
    raise ValueError('Invalid data_format:', data_format)


def assign_moving_average(variable, value, decay, name=None):
  """Compute the moving average of a variable.

  The moving average of 'variable' updated with 'value' is:
    variable * decay + value * (1 - decay)

  The returned Operation sets 'variable' to the newly computed moving average.

  The new value of 'variable' can be set with the 'AssignSub' op as:
     variable -= (1 - decay) * (variable - value)

  Args:
    variable: A Variable.
    value: A tensor with the same shape as 'variable'.
    decay: A float Tensor or float value.  The moving average decay.
    name: Optional name of the returned operation.

  Returns:
    A reference to the input 'variable' tensor with the newly computed
    moving average.
  """
  with tf.name_scope(name, "AssignMovingAvg",
                      [variable, value, decay]) as scope:
    with tf.colocate_with(variable):
      decay = tf.convert_to_tensor(1.0 - decay, name="decay")
      if decay.dtype != variable.dtype.base_dtype:
        decay = tf.cast(decay, variable.dtype.base_dtype)
      else:
        update_delta = (variable - value) * decay
      return tf.assign_sub(variable, update_delta, name=scope)


def static_cond(pred, fn1, fn2):
  """Return either fn1() or fn2() based on the boolean value of `pred`.

  Same signature as `control_flow_ops.cond()` but requires pred to be a bool.

  Args:
    pred: A value determining whether to return the result of `fn1` or `fn2`.
    fn1: The callable to be performed if pred is true.
    fn2: The callable to be performed if pred is false.

  Returns:
    Tensors returned by the call to either `fn1` or `fn2`.

  Raises:
    TypeError: if `fn1` or `fn2` is not callable.
  """
  if not callable(fn1):
    raise TypeError('fn1 must be callable.')
  if not callable(fn2):
    raise TypeError('fn2 must be callable.')
  if pred:
    return fn1()
  else:
    return fn2()


def smart_cond(pred, fn1, fn2, name=None):
  """Return either fn1() or fn2() based on the boolean predicate/value `pred`.

  If `pred` is bool or has a constant value it would use `static_cond`,
  otherwise it would use `tf.cond`.

  Args:
    pred: A scalar determining whether to return the result of `fn1` or `fn2`.
    fn1: The callable to be performed if pred is true.
    fn2: The callable to be performed if pred is false.
    name: Optional name prefix when using tf.cond
  Returns:
    Tensors returned by the call to either `fn1` or `fn2`.
  """
  pred_value = constant_value(pred)
  if pred_value is not None:
    # Use static_cond if pred has a constant value.
    return static_cond(pred_value, fn1, fn2)
  else:
    # Use dynamic cond otherwise.
    return tf.cond(pred, fn1, fn2, name)


def constant_value(value_or_tensor_or_var, dtype=None):
  """Returns value if value_or_tensor_or_var has a constant value.

  Args:
    value_or_tensor_or_var: A value, a `Tensor` or a `Variable`.
    dtype: Optional `tf.dtype`, if set it would check it has the right
      dtype.

  Returns:
    The constant value or None if it not constant.

  Raises:
    ValueError: if value_or_tensor_or_var is None or the tensor_variable has the
    wrong dtype.
  """
  if value_or_tensor_or_var is None:
    raise ValueError('value_or_tensor_or_var cannot be None')
  value = value_or_tensor_or_var
  if isinstance(value_or_tensor_or_var, (tf.Tensor, tf.Variable)):
    if dtype and value_or_tensor_or_var.dtype != dtype:
      raise ValueError('It has the wrong type %s instead of %s' % (
          value_or_tensor_or_var.dtype, dtype))
    if isinstance(value_or_tensor_or_var, tf.Variable):
      value = None
    else:
      value = tf.contrib.util.constant_value(value_or_tensor_or_var)
  return value
