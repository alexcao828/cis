import tensorflow as tf
import numpy as np

class imitate:
    def __init__(self, policy, num_time_steps, mu):
        """
        param policy: network
        num_time_steps: number of time steps (padded & truncated) per review
        mu: subtracted time penalty
        """

        self.policy = policy
        self.num_time_steps = num_time_steps

        lr = 1e-3
        num_classes = 2
        self.num_actions = 2

        # inputs for train_op
        self.y = tf.placeholder(tf.float32, [None, num_classes])
        y_tiled = tf.tile(self.y, [1, self.num_time_steps])
        y_tiled = tf.reshape(y_tiled, [-1, self.num_time_steps, num_classes])

        self.Ts = tf.placeholder(tf.int32, [None, 1])

        self.ideal_policies = tf.placeholder(tf.float32, [None, self.num_time_steps, self.num_actions])

        class_probs = self.policy.class_probs
        cross_entropies = -tf.reduce_sum(y_tiled * tf.log(1e-10+class_probs), axis=2)
        self.cross_entropy = tf.reduce_mean(cross_entropies)

        batch_size = tf.shape(self.Ts)[0]
        gatherer = tf.range(batch_size)
        gatherer = tf.reshape(gatherer, [batch_size, 1])
        gatherer = tf.concat([gatherer, self.Ts], 1)
        class_probs_T = tf.gather_nd(class_probs, gatherer)
        self.guesses = tf.math.argmax(class_probs_T, 1)

        self.rewards_acc = -cross_entropies
        rewards_time = mu * tf.dtypes.cast(tf.range(1, num_time_steps+1), tf.float32)
        self.rewards = self.rewards_acc - rewards_time

        self.act_probs = self.policy.act_probs
        imitations = -tf.reduce_sum(self.ideal_policies * tf.log(1e-10 + self.act_probs), axis=2)
        self.imitation = tf.reduce_mean(imitations)

        loss = self.imitation + self.cross_entropy
        self.train_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        self.actProbsOut = self.policy.act_probs[:, :, 1]
        self.classProbsOut = self.policy.class_probs[:, :, 1]

    def _make_ideal_policies(self, rewards, rewards_acc):
        cs = np.argmax(rewards, 1)
        batch_size = len(cs)

        ideal_policies = np.zeros((batch_size, self.num_time_steps , self.num_actions))
        rs_acc = np.zeros(batch_size)
        for i in range(batch_size):
            temp = np.zeros((self.num_time_steps, self.num_actions))

            c = cs[i]
            temp[0:c, 0] = 1.0
            temp[c:self.num_time_steps, 1] = 1.0

            ideal_policies[i, :, :] = temp
            rs_acc[i] = rewards_acc[i, c]

        cs = cs.reshape(batch_size, 1)
        return ideal_policies, cs, rs_acc

    def _run_policies(self, act_probs, rewards_acc):
        batch_size = np.shape(act_probs)[0]

        Ts = np.zeros(batch_size)
        rs_acc = np.zeros(batch_size)
        for i in range(batch_size):
            policy = act_probs[i, :, :]

            T = 0
            complete = False
            # Run through each "episode"
            while complete == False:
                action = np.argmax(policy[T])
                T += 1
                if action == 1 or T == self.num_time_steps:
                    complete = True

            Ts[i] = T
            rs_acc[i] = rewards_acc[i, T-1]

        Ts = Ts.reshape(batch_size, 1)
        rs_acc = rs_acc.reshape(batch_size, 1)
        return Ts, rs_acc

    def update(self, tokens, y):
        rewards, rewards_acc = tf.get_default_session().run((self.rewards, self.rewards_acc), feed_dict={self.policy.tokens: tokens, self.y: y})

        ideal_policies, Ts, rs_acc = self._make_ideal_policies(rewards, rewards_acc)

        _, guesses, ce, i = tf.get_default_session().run((self.train_opt, self.guesses, self.cross_entropy, self.imitation),
                                                         feed_dict={self.policy.tokens: tokens, self.y: y, self.Ts: Ts,
                                                                    self.ideal_policies: ideal_policies})

        return guesses, ce, i, rs_acc, Ts

    def eval(self, tokens, y):
        rewards, reward_acc, act_probs = tf.get_default_session().run((self.rewards, self.rewards_acc, self.act_probs), feed_dict={self.policy.tokens: tokens, self.y: y})

        ideal_policies, _, _ = self._make_ideal_policies(rewards, reward_acc)
        Ts, rs_acc = self._run_policies(act_probs, reward_acc)

        guesses, ce, i = tf.get_default_session().run((self.guesses, self.cross_entropy, self.imitation),
                                                      feed_dict={self.policy.tokens: tokens, self.y: y, self.Ts: Ts,
                                                                 self.ideal_policies: ideal_policies})

        actProbsOut, classProbsOut = tf.get_default_session().run((self.actProbsOut, self.classProbsOut),
                                                                      feed_dict={self.policy.tokens: tokens, self.y: y})

        return guesses, ce, i, rs_acc, Ts, actProbsOut, classProbsOut
