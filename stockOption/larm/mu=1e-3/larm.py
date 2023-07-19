import tensorflow as tf
import numpy as np

class larm:
    def __init__(self, policy, num_time_steps, mu):
        """
        param policy: network
        num_time_steps: number of time steps in stocks
        mu: subtracted time penalty
        """

        self.policy = policy
        self.num_time_steps = num_time_steps

        lr = 1e-4
        num_classes = 2
        self.num_actions = 2

        # inputs for train_op
        self.y = tf.placeholder(tf.float32, [None, num_classes])
        y_tiled = tf.tile(self.y, [1, self.num_time_steps])
        y_tiled = tf.reshape(y_tiled, [-1, self.num_time_steps, num_classes])

        self.class_probs = self.policy.class_probs
        class_probs_loss = tf.reduce_sum(y_tiled * self.class_probs, axis=2)

        self.act_probs = self.policy.act_probs
        self.act_probs = self.act_probs[:, :, 1]
        A = tf.nn.dropout(self.act_probs, keep_prob=0.6)*0.6
        B = tf.math.cumprod(1-A, axis=1, exclusive=True)
        A = A*B

        loss = tf.reduce_mean(-tf.log(1e-10+tf.reduce_sum(class_probs_loss*A, axis=1)))
        loss = loss + mu * tf.reduce_mean(tf.reduce_sum(tf.dtypes.cast(tf.range(1,num_time_steps+1), tf.float32) * A, 1))
        self.train_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        self.actProbsOut = self.policy.act_probs[:, :, 1]

    def _run_policies(self, act_probs, class_probs, y):
        batch_size = np.shape(act_probs)[0]

        Ts = np.zeros(batch_size)
        rs_acc = np.zeros(batch_size)
        guess = np.zeros(batch_size)
        for i in range(batch_size):
            policy = act_probs[i, :]

            T = 0
            complete = False
            # Run through each "episode"
            while complete == False:
                action = policy[T]
                T += 1
                if (np.random.uniform() < action) or (T == self.num_time_steps):
                    complete = True

            Ts[i] = T
            rs_acc[i] = np.log(class_probs[i, int(T-1), np.argmax(y[i])])
            guess[i] = np.argmax(class_probs[i,int(T-1), :])

        return Ts, rs_acc, guess

    def update(self, stocks, y):
        _, act_probs, class_probs = tf.get_default_session().run((self.train_opt, self.act_probs, self.class_probs),
                                                         feed_dict={self.policy.stocks: stocks, self.y: y})

        Ts, rs_acc, guess = self._run_policies(act_probs, class_probs, y)

        return Ts, rs_acc, guess

    def getActProbs(self, stocks):
        actProbsOut = tf.get_default_session().run((self.actProbsOut), feed_dict={self.policy.stocks: stocks})
        return actProbsOut


    def _run_oneMonth_policy(self, stocks):
        stocks = stocks.reshape(1, self.num_time_steps, -1)
        act_probs, class_probs = tf.get_default_session().run((self.policy.act_probs, self.policy.class_probs),
                                                              feed_dict={self.policy.stocks: stocks})
        act_probs = np.squeeze(act_probs)
        class_probs = np.squeeze(class_probs)

        action_space = np.arange(self.num_actions)
        T = 0
        complete = False
        # Run through each "episode"
        while complete == False:
            temp = act_probs[T]
            action = np.random.choice(action_space, p=temp)
            T += 1
            if action == 1 or T == self.num_time_steps:
                complete = True
        class_choice = np.argmax(class_probs[int(T - 1)])
        return T, class_choice

    def _calc_profit(self, close, choice):
        day1Price = close[0]
        day30Price = close[self.num_time_steps - 1]
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

    def evalYear(self, stocks, close):
        Ts, corrects, profits = self._run_oneYear_policy(stocks, close)
        return Ts, corrects, profits

