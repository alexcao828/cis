import tensorflow as tf
import numpy as np

class imitate:
    def __init__(self, policy, num_time_steps, num_features, mu):
        """
        param policy: network
        num_time_steps: number of time steps in option
        num_features:
        mu: subtracted time penalty
        """

        self.policy = policy
        self.num_time_steps = num_time_steps
        self.num_features = num_features

        lr = 1e-4
        num_classes = 2
        self.num_actions = 2

        # inputs for train_op
        self.y = tf.placeholder(tf.float32, [None, num_classes])
        y_tiled = tf.tile(self.y, [1, self.num_time_steps])
        y_tiled = tf.reshape(y_tiled, [-1, self.num_time_steps, num_classes])

        self.Ts = tf.placeholder(tf.int32, [None, 1])

        self.ideal_policies = tf.placeholder(tf.float32, [None, self.num_time_steps, self.num_actions])

        self.class_probs = self.policy.class_probs
        cross_entropies = -tf.reduce_sum(y_tiled * tf.log(1e-10+self.class_probs), axis=2)
        self.cross_entropy = tf.reduce_mean(cross_entropies)

        batch_size = tf.shape(self.Ts)[0]
        gatherer = tf.range(batch_size)
        gatherer = tf.reshape(gatherer, [batch_size, 1])
        gatherer = tf.concat([gatherer, self.Ts], 1)
        class_probs_T = tf.gather_nd(self.class_probs, gatherer)
        self.guesses = tf.math.argmax(class_probs_T, 1)

        self.rewards_acc = -cross_entropies
        rewards_time = mu * tf.dtypes.cast(tf.range(1, num_time_steps+1), tf.float32)
        self.rewards = self.rewards_acc - rewards_time

        self.act_probs = self.policy.act_probs
        imitations = -tf.reduce_sum(self.ideal_policies * tf.log(1e-10 + self.act_probs), axis=2)
        self.imitation = tf.reduce_mean(imitations)

        loss = self.imitation + self.cross_entropy
        self.train_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        self.actProbsOut = self.policy.act_probs[:,:,1]

    def _make_ideal_policies(self, rewards):
        cs = np.argmax(rewards, 1)
        batch_size = len(cs)

        ideal_policies = np.zeros((batch_size, self.num_time_steps , self.num_actions))
        for i in range(batch_size):
            temp = np.zeros((self.num_time_steps, self.num_actions))

            c = cs[i]
            temp[0:c, 0] = 1.0
            temp[c:self.num_time_steps, 1] = 1.0

            ideal_policies[i, :, :] = temp
        cs = cs.reshape(batch_size, 1)
        return ideal_policies, cs

    def _run_oneMonth_policy(self, stocks):
        stocks = stocks.reshape(1, self.num_time_steps, -1)
        act_probs, class_probs = tf.get_default_session().run((self.act_probs, self.class_probs), feed_dict={self.policy.stocks: stocks})
        policy = act_probs[0, :, :]
        T = 0
        complete = False
        # Run through each "episode"
        while complete == False:
            action = np.argmax(policy[T])
            T += 1
            if action == 1 or T == self.num_time_steps:
                complete = True
        class_probs = class_probs[0, :, :]
        class_choice = np.argmax(class_probs[int(T-1)])
        return T, class_choice

    def _calc_profit(self, close, choice):
        day1Price = close[0]
        day30Price = close[self.num_time_steps-1]
        if day30Price >= day1Price:
            true_y = 1
        else:
            true_y = 0
        if choice == true_y:
            correct = 1
            profit = np.abs(day30Price - day1Price)
        else:
            correct = 0
            profit = -np.abs(day30Price - day1Price)
        return correct, profit

    def _run_oneYear_policy(self, stocks, close):
        Ts = []
        corrects = []
        profits = []
        while np.shape(stocks)[0] >= self.num_time_steps:
            input = stocks[0:self.num_time_steps]
            T, choice = self._run_oneMonth_policy(input)
            correct, profit = self._calc_profit(close, choice)
            Ts.append(T)
            corrects.append(correct)
            profits.append(profit)
            stocks = stocks[T:]
            close = close[T:]
        return Ts, corrects, profits

    def update(self, stocks, y):
        rewards = tf.get_default_session().run((self.rewards), feed_dict={self.policy.stocks: stocks, self.y: y})
        ideal_policies, Ts = self._make_ideal_policies(rewards)
        _, guesses = tf.get_default_session().run((self.train_opt, self.guesses),
                                                         feed_dict={self.policy.stocks: stocks, self.y: y, self.Ts: Ts,
                                                                    self.ideal_policies: ideal_policies})
        return guesses, Ts

    def getActProbs(self, stocks):
        actProbsOut = tf.get_default_session().run((self.actProbsOut), feed_dict={self.policy.stocks: stocks})
        return actProbsOut

    def evalYear(self, stocks, close):
        Ts, corrects, profits = self._run_oneYear_policy(stocks, close)
        return Ts, corrects, profits
