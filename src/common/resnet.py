#!/usr/bin/env python
#taken from https://github.com/raghakot/keras-resnet/blob/master/resnet.py

from __future__ import division

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.core import Lambda
from keras.layers.merge import (dot, concatenate)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

NUM_EMBEDDING = 512 #256 #512 #1024 #256 #1024 #256
TOP_HIDDEN = 4 #1 #4
NORMALIZATION_ON = False #True #False #True

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier

def _bn_relu_for_dense(input):
    norm = BatchNormalization(axis=1)(input)
    return Activation('relu')(norm)

def _top_network(input):
    raw_result = _bn_relu_for_dense(input)
    for _ in xrange(TOP_HIDDEN):
        raw_result = Dense(units=NUM_EMBEDDING, kernel_initializer='he_normal')(raw_result)
        raw_result = _bn_relu_for_dense(raw_result)
    output = Dense(units=2, activation='softmax', kernel_initializer='he_normal')(raw_result)
    return output

class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, is_classification):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        last_activation = None
        if is_classification:
            last_activation = "softmax"
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation=last_activation)(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, is_classification=True):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2], is_classification)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])

    @staticmethod
    def build_top_network(edge_model):
        number_of_top_layers = 3 + TOP_HIDDEN * 3
        input = Input(shape=(2 * NUM_EMBEDDING,))
        output = edge_model.layers[-number_of_top_layers](input) #_top_network(input)
        for index in xrange(-number_of_top_layers + 1, 0):
            output = edge_model.layers[index](output)
        return Model(inputs=input, outputs=output)

    @staticmethod
    def build_bottom_network(edge_model, input_shape):
        channels, height, width = input_shape
        input = Input(shape=(height, width, channels))
        branch = edge_model.layers[3]
        output = branch(input)
        if NORMALIZATION_ON:
            output = Lambda(lambda x: K.l2_normalize(x, axis=1))(output) 
        return Model(inputs=input, outputs=output)

    @staticmethod
    def build_siamese_resnet_18(input_shape, num_outputs):
        channels, height, width = input_shape
        branch_channels = 3 #channels / 2
        branch_input_shape = (branch_channels, height, width)
        branch = ResnetBuilder.build_resnet_18(branch_input_shape, NUM_EMBEDDING, False)
        input = Input(shape=(height, width, channels))
        first_branch = branch(Lambda(lambda x: x[:, :, :, :3])(input))
        second_branch = branch(Lambda(lambda x: x[:, :, :, 3:])(input))
        if NORMALIZATION_ON:
            first_branch = Lambda(lambda x: K.l2_normalize(x, axis=1))(first_branch)
            second_branch = Lambda(lambda x: K.l2_normalize(x, axis=1))(second_branch) 
        
        raw_result = concatenate([first_branch, second_branch])
        output = _top_network(raw_result)
        
        # raw_result = dot([first_branch, second_branch], axes=1)
        # result = Lambda(lambda x: (K.clip(x, 0.5, 1) - 0.5) * 2.0)(raw_result)
        # negated_result = Lambda(lambda x: 1 - x)(result)
        # output = concatenate([negated_result, result])
        
        return Model(inputs=input, outputs=output)

    @staticmethod
    def build_pixel_comparison_network(input_shape):
        channels, height, width = input_shape
        input = Input(shape=(height, width, channels))
        first = Flatten()(Lambda(lambda x: x[:, :, :, :1])(input))
        second = Flatten()(Lambda(lambda x: x[:, :, :, 1:])(input))
        # second = Lambda(lambda x: -x)(second)
        # difference = add([first, second])
        # raw_result = Lambda(lambda x: K.mean(K.abs(x), axis=1, keepdims=True))(difference)
        # prob_zero = Lambda(lambda x: x / 255.0)(raw_result)
        # prob_one = Lambda(lambda x: 1.0 - x)(prob_zero)
        prob_one = dot([first, second], axes=1, normalize=True)
        prob_zero = Lambda(lambda x: 1.0 - x)(prob_one)
        output = concatenate([prob_zero, prob_one])
        return Model(inputs=input, outputs=output)

        