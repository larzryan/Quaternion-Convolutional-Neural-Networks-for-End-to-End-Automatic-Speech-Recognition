# Created this file to pull in the old backend code required for this to work.
# This is pulled from https://github.com/keras-team/keras/blob/v3.3.3/keras/src/legacy/backend.py#L479-L508

import tensorflow as tf

py_all = all


def is_sparse(tensor):
    spec = getattr(tensor, "_type_spec", None)
    if spec is not None:
        return isinstance(spec, tf.SparseTensorSpec)
    return isinstance(tensor, tf.SparseTensor)


def to_dense(tensor):
    if is_sparse(tensor):
        return tf.sparse.to_dense(tensor)
    else:
        return tensor


def concatenate(tensors, axis=-1):
    if axis < 0:
        rank = (tensors[0]).ndim
        if rank:
            axis %= rank
        else:
            axis = 0

    if py_all(is_sparse(x) for x in tensors):
        return tf.compat.v1.sparse_concat(axis, tensors)
    elif py_all(isinstance(x, tf.RaggedTensor) for x in tensors):
        return tf.concat(tensors, axis)
    else:
        return tf.concat([to_dense(x) for x in tensors], axis)


def dot(x, y):
    if x.ndim is not None and (x.ndim > 2 or y.ndim > 2):
        x_shape = []
        for i, s in zip(x.shape, tf.unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(y.shape, tf.unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)
        y_shape = tuple(y_shape)
        y_permute_dim = list(range(y.ndim))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(
            tf.matmul(xt, yt), x_shape[:-1] + y_shape[:-2] + y_shape[-1:]
        )
    if is_sparse(x):
        out = tf.sparse.sparse_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out


def _preprocess_conv1d_input(x, data_format):
    tf_data_format = "NWC"  # to pass TF Conv2dNative operations
    if data_format == "channels_first":
        tf_data_format = "NCW"
    return x, tf_data_format


def _preprocess_conv2d_input(x, data_format, force_transpose=False):
    tf_data_format = "NHWC"
    if data_format == "channels_first":
        if force_transpose:
            x = tf.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        else:
            tf_data_format = "NCHW"
    return x, tf_data_format


def _preprocess_conv3d_input(x, data_format):
    tf_data_format = "NDHWC"
    if data_format == "channels_first":
        tf_data_format = "NCDHW"
    return x, tf_data_format


def _preprocess_padding(padding):
    if padding == "same":
        padding = "SAME"
    elif padding == "valid":
        padding = "VALID"
    else:
        raise ValueError(f"Invalid padding: {padding}")
    return padding


def temporal_padding(x, padding=(1, 1)):
    assert len(padding) == 2
    pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
    return tf.compat.v1.pad(x, pattern)


def conv1d(
    x, kernel, strides=1, padding="valid", data_format=None, dilation_rate=1
):
    if data_format is None:
        data_format = tf.keras.config.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown data_format: {data_format}")

    kernel_shape = kernel.shape.as_list()
    if padding == "causal":
        # causal (dilated) convolution:
        left_pad = dilation_rate * (kernel_shape[0] - 1)
        x = temporal_padding(x, (left_pad, 0))
        padding = "valid"
    padding = _preprocess_padding(padding)

    x, tf_data_format = _preprocess_conv1d_input(x, data_format)
    x = tf.compat.v1.nn.convolution(
        input=x,
        filter=kernel,
        dilation_rate=dilation_rate,
        strides=strides,
        padding=padding,
        data_format=tf_data_format,
    )
    if data_format == "channels_first" and tf_data_format == "NWC":
        x = tf.transpose(x, (0, 2, 1))  # NWC -> NCW
    return x


def conv2d(
    x,
    kernel,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
):
    if data_format is None:
        data_format = tf.keras.config.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown data_format: {data_format}")

    x, tf_data_format = _preprocess_conv2d_input(x, data_format)
    padding = _preprocess_padding(padding)
    x = tf.compat.v1.nn.convolution(
        input=x,
        filter=kernel,
        dilation_rate=dilation_rate,
        strides=strides,
        padding=padding,
        data_format=tf_data_format,
    )
    if data_format == "channels_first" and tf_data_format == "NHWC":
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


def conv2d_transpose(
    x,
    kernel,
    output_shape,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
):
    if data_format is None:
        data_format = tf.keras.config.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown data_format: {data_format}")

    # `atrous_conv2d_transpose` only supports NHWC format, even on GPU.
    if data_format == "channels_first" and dilation_rate != (1, 1):
        force_transpose = True
    else:
        force_transpose = False

    x, tf_data_format = _preprocess_conv2d_input(
        x, data_format, force_transpose
    )

    if data_format == "channels_first" and tf_data_format == "NHWC":
        output_shape = (
            output_shape[0],
            output_shape[2],
            output_shape[3],
            output_shape[1],
        )
    if output_shape[0] is None:
        output_shape = (tf.shape(x)[0],) + tuple(output_shape[1:])

    if isinstance(output_shape, (tuple, list)):
        output_shape = tf.stack(list(output_shape))

    padding = _preprocess_padding(padding)
    if tf_data_format == "NHWC":
        strides = (1,) + strides + (1,)
    else:
        strides = (1, 1) + strides

    if dilation_rate == (1, 1):
        x = tf.compat.v1.nn.conv2d_transpose(
            x,
            kernel,
            output_shape,
            strides,
            padding=padding,
            data_format=tf_data_format,
        )
    else:
        if dilation_rate[0] != dilation_rate[1]:
            raise ValueError(
                "Expected the 2 dimensions of the `dilation_rate` argument "
                "to be equal to each other. "
                f"Received: dilation_rate={dilation_rate}"
            )
        x = tf.nn.atrous_conv2d_transpose(
            x, kernel, output_shape, rate=dilation_rate[0], padding=padding
        )
    if data_format == "channels_first" and tf_data_format == "NHWC":
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


def conv3d(
    x,
    kernel,
    strides=(1, 1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1, 1),
):
    if data_format is None:
        data_format = tf.keras.config.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown data_format: {data_format}")

    x, tf_data_format = _preprocess_conv3d_input(x, data_format)
    padding = _preprocess_padding(padding)
    x = tf.compat.v1.nn.convolution(
        input=x,
        filter=kernel,
        dilation_rate=dilation_rate,
        strides=strides,
        padding=padding,
        data_format=tf_data_format,
    )
    if data_format == "channels_first" and tf_data_format == "NDHWC":
        x = tf.transpose(x, (0, 4, 1, 2, 3))
    return x


def bias_add(x, bias, data_format=None):
    if data_format is None:
        data_format = tf.keras.config.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown data_format: {data_format}")
    bias_shape = bias.shape
    if len(bias_shape) != 1 and len(bias_shape) != x.ndim - 1:
        raise ValueError(
            f"Unexpected bias dimensions {len(bias_shape)}. "
            f"Expected it to be 1 or {x.ndim - 1} dimensions"
        )

    if len(bias_shape) == 1:
        if data_format == "channels_first":
            return tf.nn.bias_add(x, bias, data_format="NCHW")
        return tf.nn.bias_add(x, bias, data_format="NHWC")
    if x.ndim in (3, 4, 5):
        if data_format == "channels_first":
            bias_reshape_axis = (1, bias_shape[-1]) + bias_shape[:-1]
            return x + tf.reshape(bias, bias_reshape_axis)
        return x + tf.reshape(bias, (1,) + bias_shape)
    return tf.nn.bias_add(x, bias)
