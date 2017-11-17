import tensorflow as tf

class GRUCell(tf.nn.rnn_cell.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units):
        self._num_units = num_units

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                ru = tf.nn.rnn_cell._linear([inputs, state],
                                        2 * self._num_units, True, 1.0)
                ru = tf.nn.sigmoid(ru)
                r, u = tf.split(1, 2, ru)
            with tf.variable_scope("Candidate"):
                c = tf.nn.tanh(tf.nn.rnn_cell._linear([inputs, r * state],
                                             self._num_units, True))
            new_h = u * state + (1 - u) * c
        return new_h, new_h