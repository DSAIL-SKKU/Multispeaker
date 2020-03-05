import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.layers import core
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper \
        import _bahdanau_score, _BaseAttentionMechanism, BahdanauAttention, \
               AttentionWrapper, AttentionWrapperState


def get_embed(inputs, num_inputs, embed_size, name):
    embed_table = tf.get_variable(
            name, [num_inputs, embed_size], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.1))
    return tf.nn.embedding_lookup(embed_table, inputs)


def prenet(inputs, is_training, layer_sizes, drop_prob, scope=None):
	"""
	Args:
		inputs: input vector
		is_training: dropout option
		layer_sizes: iteration number
	
	Output:
		x: prenet
	"""
	x = inputs
	drop_rate = drop_prob if is_training else 0.0 # set dropout rate 0.5 (only training)
	with tf.variable_scope(scope or 'prenet'):
		for i, size in enumerate(layer_sizes): # iterate layer_sizes
			dense = tf.layers.dense(x, units=size, activation=tf.nn.relu, name='dense_%d' % (i + 1))
			x = tf.layers.dropout(dense, rate=drop_rate, training=is_training, name='dropout_%d' % (i + 1)) 
	return x


def encoder_cbhg(inputs, input_lengths, is_training, depth):
	"""
	Args:
		inputs: input tensor
		input_lengths: length of input tensor
		is_training: Batch Normalization option in Conv1D
		depth: dimensionality option of Highway net and Bidirectical GRU's output
	
	Output:
		cbhg function
	"""
	input_channels = inputs.get_shape()[2] # 3rd element of inputs' shape
	return cbhg(
		inputs,
		input_lengths,
		is_training,
		scope='encoder_cbhg',
		K=16,
		projections=[128, input_channels],
		depth=depth)


def post_cbhg(inputs, input_dim, is_training, depth):
	"""
	Args:
		inputs: input tensor
		input_dim: dimension of input tensor
		is_training: Batch Normalization option in Conv1D
		depth: dimensionality option of Highway net and Bidirectical GRU's output
	
	Output:
		cbhg function
	"""
	return cbhg(
		inputs,
		None,
		is_training,
		scope='post_cbhg',
		K=8,
		projections=[256, input_dim],
		depth=depth)


def cbhg(inputs, input_lengths, is_training, bank_size, bank_channel_size, 
		maxpool_width, highway_depth, rnn_size, proj_sizes, proj_width, scope, 
		before_highway = None, encoder_rnn_init_state = None):
    """
    Args:
        inputs: input tensor
        input_lengths: length of input tensor
        is_training: Batch Normalization option in Conv1D
        scope: network or model name
        K: kernel size range
        projections: projection layers option
        depth: dimensionality option of Highway net and Bidirectical GRU's output
    The layers in the code are staked in the order in which they came out.
    """

    batch_size = tf.shape(inputs)[0]
    with tf.variable_scope(scope):
        with tf.variable_scope('conv_bank'):

            conv_outputs = tf.concat(
                [conv1d(inputs, k, 128, tf.nn.relu, is_training, 'conv1d_%d' % k) for k in range(1, bank_size + 1)], #1D Convolution layers using multiple types of Convolution Kernel.
                axis=-1																						 #Iterate K with increasing filter size by 1.
            )# Convolution bank: concatenate on the last axis to stack channels from all convolutions

        # Maxpooling:
        maxpool_output = tf.layers.max_pooling1d(
            conv_outputs,
            pool_size=maxpool_width,
            strides=1,
            padding='same') #1D Maxpooling layer(strides=1, width=2) 

        # Two projection layers:
        proj1_output = conv1d(maxpool_output, proj_width, projections[0], tf.nn.relu, is_training, 'proj_1')#1st Conv1D projections
        proj2_output = conv1d(proj1_output, proj_width, projections[1], None, is_training, 'proj_2')#2nd Conv1D projections

        # Residual connection:
        if before_highway is not None:
            expanded_before_highway = tf.expand_dims(before_highway, [1])
            tiled_before_highway = tf.tile(
                    expanded_before_highway, [1, tf.shape(proj2_out)[1], 1])
            highway_input = proj2_out + inputs + tiled_before_highway
        
        else:
            highway_input = proj2_out + inputs

        # Handle dimensionality mismatch:
        if highway_input.shape[2] != rnn_size:
            highway_input = tf.layers.dense(highway_input, rnn_size)

        # 4-layer HighwayNet:
        for idx in range(highway_depth):
            highway_input = highwaynet(highway_input, 'highway_%d' % (idx+1)) #make 4 Highway net layers
        rnn_input = highway_input

        # Bidirectional RNN
        if encoder_rnn_init_state is not None:
            initial_state_fw, initial_state_bw = tf.split(encoder_rnn_init_state, 2, 1)
        else:
            initial_state_fw, initial_state_bw = None, None

        outputs, states = tf.nn.bidirectional_dynamic_rnn( #make Bidirectional GRU
            GRUCell(rnn_size),
            GRUCell(rnn_size),
            rnn_input,
            sequence_length=input_lengths,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            dtype=tf.float32)
        return tf.concat(outputs, axis=2)  # Concat forward sequence and backward sequence


def batch_tile(tensor, batch_size):
    expaneded_tensor = tf.expand_dims(tensor, [0])
    return tf.tile(expaneded_tensor, \
            [batch_size] + [1 for _ in tensor.get_shape()])


def highwaynet(inputs, scope):
	highway_dim = int(inputs.get_shape()[-1])

	with tf.variable_scope(scope):
		H = tf.layers.dense(
			inputs,
			units=highway_dim,
			activation=tf.nn.relu,
			name='H')
		T = tf.layers.dense(
			inputs,
			units=highway_dim,
			activation=tf.nn.sigmoid,
			name='T',
			bias_initializer=tf.constant_initializer(-1.0))
		return H * T + inputs * (1.0 - T)


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
	"""
	Args:
		inputs: input tensor
		kernel_size: length of the 1D convolution window
		channels: dimensionality of the output space
		activation: Activation function (None means linear activation)
		is_training: Batch Normalization option in Conv1D
		scope: namespace
	
	Output:
		output tensor
	"""
	with tf.variable_scope(scope):
		conv1d_output = tf.layers.conv1d( # creates a convolution kernel
			inputs,
			filters=channels,
			kernel_size=kernel_size,
			activation=activation,
			padding='same') # return output tensor
		return tf.layers.batch_normalization(conv1d_output, training=is_training)
