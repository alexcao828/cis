import tensorflow as tf

class policy_net:
    def __init__(self, num_time_steps, num_frequencies, num_leads):
        """
        num_time_steps: number of time steps in spectrogram
        num_frequencies: number of frequencies in spectrogram
        num_leads: number of leads in ecg
        """

        self.num_time_steps = num_time_steps
        self.num_frequencies = num_frequencies
        self.num_leads = num_leads

        self.lstm_dim = 128
        self.fc_hidden_dim = 64

        self.num_classes = 5
        self.num_actions = 2

        self.weight_init = tf.contrib.layers.xavier_initializer()

        self.spectrograms = tf.placeholder(tf.float32, [None, self.num_time_steps, self.num_frequencies, self.num_leads])

        self._predict()

    def _fully_connected(self, x, units):
        x = tf.layers.dense(x, units=units, use_bias=True, kernel_initializer=self.weight_init)
        return x

    def _predict(self):
        x = tf.reshape(self.spectrograms, [-1, self.num_time_steps, self.num_frequencies*self.num_leads])
        cell = tf.nn.rnn_cell.LSTMCell(self.lstm_dim, state_is_tuple=True)
        val, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

        val = tf.reshape(val, [-1, self.lstm_dim])

        x1 = self._fully_connected(val, units=self.fc_hidden_dim)
        x1 = tf.nn.relu(x1)
        x1 = self._fully_connected(x1, units=self.num_classes)
        self.class_probs = tf.nn.softmax(x1)
        self.class_probs = tf.reshape(self.class_probs, [-1, self.num_time_steps, self.num_classes])

        x2 = self._fully_connected(val, units=self.fc_hidden_dim)
        x2 = tf.nn.relu(x2)
        x2 = self._fully_connected(x2, units=self.num_actions)
        self.act_probs = tf.nn.softmax(x2, 1)
        self.act_probs = tf.reshape(self.act_probs, [-1, self.num_time_steps, self.num_actions])
