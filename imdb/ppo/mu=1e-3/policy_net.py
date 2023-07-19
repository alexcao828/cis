import tensorflow as tf
import numpy as np

class policy_net:
    def __init__(self, name: str, vocab_size, num_time_steps):
        """
        param name: string
        param vocab_size: total number of unique words in training corpus + 1
        num_time_steps: number of time steps (padded & truncated) per review
        """

        self.name = name
        self.vocab_size = vocab_size
        self.num_time_steps = num_time_steps

        K = 10.0
        self.word_embedding_dim = 32
        self.lstm_dim = 64
        self.fc_hidden_dim = 32

        self.num_classes = 2
        self.num_actions = 2

        self.weight_init = tf.contrib.layers.xavier_initializer()

        self.tokens = tf.placeholder(tf.int32, [self.num_time_steps])

        self.start_explore = np.zeros((num_time_steps, self.num_actions))
        for i in range(num_time_steps):
            self.start_explore[i, 0] = K
        self.start_explore = tf.dtypes.cast(tf.constant(self.start_explore), tf.float32)

        self._predict()

    def _fully_connected(self, x, units, scope='fully'):
        with tf.variable_scope(scope):
            x = tf.layers.dense(x, units=units, use_bias=True, kernel_initializer=self.weight_init)
            return x

    def _predict(self):
        with tf.variable_scope(self.name):
            with tf.variable_scope('word_embeddings'):
                words_embeddings = tf.Variable(tf.random.uniform([self.vocab_size, self.word_embedding_dim], minval=-1))
                x = tf.gather(words_embeddings, self.tokens)

            with tf.variable_scope('lstm'):
                x = tf.reshape(x, [1, self.num_time_steps, self.word_embedding_dim])
                cell = tf.nn.rnn_cell.LSTMCell(self.lstm_dim, state_is_tuple=True)
                val, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
                val = tf.squeeze(val)

            with tf.variable_scope('classification'):
                x1 = self._fully_connected(val, units=self.fc_hidden_dim, scope='classification_hidden')
                x1 = tf.nn.relu(x1)
                x1 = self._fully_connected(x1, units=self.num_classes, scope='classification_logits')
                self.class_probs = tf.nn.softmax(x1, 1)

            with tf.variable_scope('actions'):
                x2 = self._fully_connected(val, units=self.fc_hidden_dim, scope='actions_hidden')
                x2 = tf.nn.relu(x2)
                x2 = self._fully_connected(x2, units=self.num_actions, scope='actions_logits')
                x2 = x2 + self.start_explore
                self.act_probs = tf.nn.softmax(x2, 1)

            # with tf.variable_scope('value'):
                # ...
                # self.v_preds = ...

            self.scope = tf.get_variable_scope().name

    def get_action_prob(self, tokens):
        return tf.get_default_session().run(self.act_probs, feed_dict = {self.tokens: tokens})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
