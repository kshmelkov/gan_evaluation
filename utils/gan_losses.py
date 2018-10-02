import tensorflow as tf
from tensorflow.python.util import tf_inspect


def hinge_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    """


    Args:
        discriminator_real_outputs: Discriminator output on real data.
        discriminator_gen_outputs: Discriminator output on generated data. Expected
        to be in the range of (-inf, inf).
        real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `discriminator_real_outputs`, and must be broadcastable to
        `discriminator_real_outputs` (i.e., all dimensions must be either `1`, or
        the same as the corresponding dimension).
        generated_weights: Same as `real_weights`, but for
        `discriminator_gen_outputs`.
        scope: The scope for the operations performed in computing the loss.
        loss_collection: collection to which this loss will be added.
        reduction: A `tf.losses.Reduction` to apply to loss.
        add_summaries: Whether or not to add summaries for the loss.

    Returns:
        A loss Tensor. The shape depends on `reduction`.
    """
    with tf.name_scope(scope, 'discriminator_hinge_loss', (
            discriminator_real_outputs, discriminator_gen_outputs,
            real_weights, generated_weights)) as scope:
        discriminator_real_outputs = tf.to_float(discriminator_real_outputs)
        discriminator_gen_outputs = tf.to_float(discriminator_gen_outputs)
        discriminator_real_outputs.shape.assert_is_compatible_with(
            discriminator_gen_outputs.shape)

        loss_on_generated = tf.losses.compute_weighted_loss(
            tf.nn.relu(1.0 + discriminator_gen_outputs), generated_weights, scope,
            loss_collection=None, reduction=reduction)
        loss_on_real = tf.losses.compute_weighted_loss(
            tf.nn.relu(1.0 - discriminator_real_outputs), real_weights, scope,
            loss_collection=None, reduction=reduction)
        loss = loss_on_generated + loss_on_real
        tf.losses.add_loss(loss, loss_collection)

        if add_summaries:
            tf.summary.scalar('discriminator_gen_hinge_loss', loss_on_generated)
            tf.summary.scalar('discriminator_real_hinge_loss', loss_on_real)
            tf.summary.scalar('discriminator_hinge_loss', loss)

        return loss


def hinge_generator_loss(
    discriminator_gen_outputs,
    weights=1.0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    """

    Args:
        discriminator_gen_outputs: Discriminator output on generated data. Expected
        to be in the range of (-inf, inf).
        weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `discriminator_gen_outputs`, and must be broadcastable to
        `discriminator_gen_outputs` (i.e., all dimensions must be either `1`, or
        the same as the corresponding dimension).
        scope: The scope for the operations performed in computing the loss.
        loss_collection: collection to which this loss will be added.
        reduction: A `tf.losses.Reduction` to apply to loss.
        add_summaries: Whether or not to add summaries for the loss.

    Returns:
        A loss Tensor. The shape depends on `reduction`.
    """
    with tf.name_scope(scope, 'generator_hinge_loss', (
            discriminator_gen_outputs, weights)) as scope:

        discriminator_gen_outputs = tf.to_float(discriminator_gen_outputs)

        loss = - discriminator_gen_outputs
        loss = tf.losses.compute_weighted_loss(
            loss, weights, scope, loss_collection, reduction)

        if add_summaries:
            tf.summary.scalar('generator_hinge_loss', loss)

        return loss


def _args_to_gan_model(loss_fn):
    """Converts a loss taking individual args to one taking a GANModel namedtuple.

    The new function has the same name as the original one.

    Args:
        loss_fn: A python function taking a `GANModel` object and returning a loss
        Tensor calculated from that object. The shape of the loss depends on
        `reduction`.

    Returns:
        A new function that takes a GANModel namedtuples and returns the same loss.
    """
    # Match arguments in `loss_fn` to elements of `namedtuple`.
    # TODO(joelshor): Properly handle `varargs` and `keywords`.
    argspec = tf_inspect.getargspec(loss_fn)
    defaults = argspec.defaults or []

    required_args = set(argspec.args[:-len(defaults)])
    args_with_defaults = argspec.args[-len(defaults):]
    default_args_dict = dict(zip(args_with_defaults, defaults))

    def new_loss_fn(gan_model, **kwargs):  # pylint:disable=missing-docstring
        def _asdict(namedtuple):
            """Returns a namedtuple as a dictionary.

            This is required because `_asdict()` in Python 3.x.x is broken in classes
            that inherit from `collections.namedtuple`. See
            https://bugs.python.org/issue24931 for more details.

            Args:
                namedtuple: An object that inherits from `collections.namedtuple`.

            Returns:
                A dictionary version of the tuple.
            """
            return {k: getattr(namedtuple, k) for k in namedtuple._fields}
        gan_model_dict = _asdict(gan_model)

        # Make sure non-tuple required args are supplied.
        args_from_tuple = set(argspec.args).intersection(set(gan_model._fields))
        required_args_not_from_tuple = required_args - args_from_tuple
        for arg in required_args_not_from_tuple:
            if arg not in kwargs:
                raise ValueError('`%s` must be supplied to %s loss function.' % (
                    arg, loss_fn.__name__))

        # Make sure tuple args aren't also supplied as keyword args.
        ambiguous_args = set(gan_model._fields).intersection(set(kwargs.keys()))
        if ambiguous_args:
            raise ValueError(
                'The following args are present in both the tuple and keyword args '
                'for %s: %s' % (loss_fn.__name__, ambiguous_args))

        # Add required args to arg dictionary.
        required_args_from_tuple = required_args.intersection(args_from_tuple)
        for arg in required_args_from_tuple:
            assert arg not in kwargs
            kwargs[arg] = gan_model_dict[arg]

        # Add arguments that have defaults.
        for arg in default_args_dict:
            val_from_tuple = gan_model_dict[arg] if arg in gan_model_dict else None
            val_from_kwargs = kwargs[arg] if arg in kwargs else None
            assert not (val_from_tuple is not None and val_from_kwargs is not None)
            kwargs[arg] = (val_from_tuple if val_from_tuple is not None else
                           val_from_kwargs if val_from_kwargs is not None else
                           default_args_dict[arg])

        return loss_fn(**kwargs)

    new_docstring = """The gan_model version of %s.""" % loss_fn.__name__
    new_loss_fn.__docstring__ = new_docstring
    new_loss_fn.__name__ = loss_fn.__name__
    new_loss_fn.__module__ = loss_fn.__module__
    return new_loss_fn


hinge_generator_loss = _args_to_gan_model(
    hinge_generator_loss)
hinge_discriminator_loss = _args_to_gan_model(
    hinge_discriminator_loss)
