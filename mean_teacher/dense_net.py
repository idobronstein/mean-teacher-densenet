import tensorflow as tf

TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))

class DenseNet():
    def __init__(self, growth_rate, total_blocks, depth, bc_mode, reduction, keep_prob, is_training, n_classes):
        self.growth_rate = growth_rate
        self.total_blocks = total_blocks
        self.depth = depth
        self.bc_mode  = bc_mode 
        self.reduction = reduction
        self.keep_prob = keep_prob
        self.is_training = is_training
        self.n_classes = n_classes
        self.first_output_features = self.growth_rate * 2 
        self.layers_per_block = (self.depth - (self.total_blocks + 1)) // self.total_blocks

    def _weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())
    
    def _weight_variable_xavier(self, shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())
    
    def _bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)
    
    def _conv2d(self, _input, out_features, kernel_size, strides=[1, 1, 1, 1], padding='SAME'):
        in_features = int(_input.get_shape()[-1])
        kernel = self._weight_variable_msra([kernel_size, kernel_size, in_features, out_features], name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output
    
    def _add_block(self,  _input, growth_rate, layers_per_block):
        output = _input
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                output = self._add_internal_layer(output, growth_rate)
        return output
    
    def _add_internal_layer(self, _input, growth_rate):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel
        if not self.bc_mode:
            comp_out = self._composite_function(_input, out_features=growth_rate, kernel_size=3)
        elif self.bc_mode:
            bottleneck_out = self._bottleneck(_input, out_features=growth_rate)
            comp_out = self._composite_function(
                bottleneck_out, out_features=growth_rate, kernel_size=3)
        # concatenate _input with out from composite function
        if TF_VERSION >= 1.0:
            output = tf.concat(axis=3, values=(_input, comp_out))
        else:
            output = tf.concat(3, (_input, comp_out))
        return output
    
    
    def _composite_function(self, _input, out_features, kernel_size=3):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function"):
            # BN
            output = self._batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self._conv2d(
                output, out_features=out_features, kernel_size=kernel_size)
            # dropout(in case of training and in case it is no 1.0)
            output = self._dropout(output)
        return output
    
    def _batch_norm(self, _input):
        output = tf.contrib.layers.batch_norm(_input, scale=True, is_training=self.is_training)
        return output
    
    def _dropout(self, _input):
        output = tf.cond(self.is_training,lambda: tf.nn.dropout(_input, self.keep_prob), lambda: _input)
        return output
    
    def _bottleneck(self, _input, out_features):
        with tf.variable_scope("bottleneck"):
            output = self._batch_norm(_input)
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            output = self._conv2d(
                output, out_features=inter_features, kernel_size=1,
                padding='VALID')
            output = self._dropout(output)
        return output
    
    def _transition_layer(self, _input):
        """Call H_l composite function with 1x1 kernel and after average
        pooling
        """
        # call composite function with 1x1 kernel
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        output = self._composite_function( _input, out_features=out_features, kernel_size=1)
        # run average pooling
        output = self._avg_pool(output, k=2)
        return output
        
    def _avg_pool(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output
    
    def _transition_layer_to_classes(self, _input):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        # BN
        output = self._batch_norm(_input)
        # ReLU
        output = tf.nn.relu(output)
        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = self._avg_pool(output, k=last_pool_kernel)
        # FC
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        W = self._weight_variable_xavier(
            [features_total, self.n_classes], name='W')
        bias = self._bias_variable([self.n_classes])
        logits = tf.matmul(output, W) + bias
        return logits
    
    
    def build(self, images):
        # first - initial 3 x 3 conv to first_output_features
        with tf.variable_scope("Initial_convolution"):
            output = self._conv2d(images, out_features=self.first_output_features, kernel_size=3)
        
        # add N required blocks
        for block in range(self.total_blocks):
            with tf.variable_scope("Block_%d" % block):
                output = self._add_block(output, self.growth_rate, self.layers_per_block)
            # last block exist without transition layer
            if block != self.total_blocks - 1:
                with tf.variable_scope("Transition_after_block_%d" % block):
                    output = self._transition_layer(output)
        
        with tf.variable_scope("Transition_to_classes"):
            logits = self._transition_layer_to_classes(output)
    
        return logits