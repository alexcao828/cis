import tensorflow as tf
import numpy as np

class ppo_train:
    def __init__(self, policy, old_policy, clip_value=0.2, c_1=1, c_2=0.01):
        """
        param policy:
        param old_policy:
        param clip_value:
        param c_1: parameter for value difference
        param c_2: parameter for entropy bonus
        """

        self.gamma = 1.0
        self.mu = 1e-3
        epsilon = 1e-5
        learning_rate = 1e-4
        num_classes = 2
        self.num_actions = 2

        self.policy = policy
        self.old_policy = old_policy
        self.num_time_steps = self.policy.num_time_steps

        self.pi_trainable = self.policy.get_trainable_variables()
        old_pi_trainable = self.old_policy.get_trainable_variables()

        # assign_operations for policy parameter values to old policy parameters
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, self.pi_trainable):
                self.assign_ops.append(tf.assign(v_old, v))

        # inputs for train_op
        with tf.variable_scope('train_input'):
            self.y = tf.placeholder(tf.float32, [num_classes])
            self.T = tf.placeholder(tf.int32, [1])
            self.T_slicer = tf.placeholder(tf.int32, [2])
            self.should_be_gae = tf.placeholder(tf.float32, [None])

        with tf.name_scope('gae/discount_reward'):
            T_squeezed = tf.squeeze(self.T)
            
            prob = self.policy.class_probs
            prob = tf.slice(prob, [T_squeezed-1, 0], [1,num_classes])
            prob = tf.reshape(prob, [num_classes])

            self.reward_acc = tf.squeeze(tf.reduce_sum(self.y * tf.log(1e-10 + prob)))

            self.time_stop = tf.dtypes.cast(T_squeezed, tf.float32)-1

            self.reward = self.reward_acc - self.mu*(self.time_stop+1)

            "maybe should use gae"
            # v_preds = self.policy.v_preds
            # v_preds = tf.squeeze(v_preds)
            # v_preds = tf.slice(v_preds, [0], [T_squeezed])

        "should use value"
        # with tf.variable_scope('loss/vf'):
        #     loss_vf = tf.squared_difference(self.rewards, v_preds)
        #     loss_vf = tf.reduce_mean(loss_vf)

        "entropy doesn't make sense?"
        # with tf.variable_scope('loss/entropy'):
        #     a_probs = tf.gather(act_probs, tf.range(T_squeezed), axis=0)
        #     entropy = -tf.reduce_sum(a_probs * tf.log(tf.clip_by_value(a_probs, 1e-10, 1.0)), axis=1)
        #     entropy = tf.reduce_mean(entropy, axis=0)  # mean of entropy of pi(obs)

        with tf.variable_scope('loss/clip'):
            act_probs = self.policy.act_probs
            act_probs_old = self.old_policy.act_probs

            first_Tminus1_selected_act_probs = tf.cond(T_squeezed > tf.constant(1), lambda: tf.reshape(
                tf.slice(act_probs, [0, 0], self.T_slicer), [T_squeezed - 1, 1]), lambda: act_probs[0, 1])

            T_selected_act_probs = act_probs[T_squeezed - 1, 1]
            T_selected_act_probs = tf.reshape(T_selected_act_probs, [1, 1])

            selected_action_prob = tf.cond(T_squeezed > tf.constant(1), lambda: tf.concat(
                [first_Tminus1_selected_act_probs, T_selected_act_probs], 0), lambda: act_probs[0, 1])
            selected_action_prob = tf.squeeze(selected_action_prob)

            first_Tminus1_selected_act_probs_old = tf.cond(T_squeezed > tf.constant(1), lambda: tf.reshape(
                tf.slice(act_probs_old, [0, 0], self.T_slicer), [T_squeezed - 1, 1]), lambda: act_probs_old[0, 1])

            T_selected_act_probs_old = act_probs_old[T_squeezed - 1, 1]
            T_selected_act_probs_old = tf.reshape(T_selected_act_probs_old, [1, 1])

            selected_action_prob_old = tf.cond(T_squeezed > tf.constant(1), lambda: tf.concat(
                [first_Tminus1_selected_act_probs_old, T_selected_act_probs_old], 0), lambda: act_probs_old[0, 1])
            selected_action_prob_old = tf.squeeze(selected_action_prob_old)

            # ratios = tf.divide(act_probs, act_probs_old)
            ratios = tf.exp(tf.log(1e-10 + selected_action_prob) - tf.log(1e-10 + selected_action_prob_old))
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
            loss_clip = tf.minimum(tf.multiply(self.should_be_gae, ratios),
                                   tf.multiply(self.should_be_gae, clipped_ratios))
            loss_clip = -tf.reduce_mean(loss_clip)

        with tf.name_scope('cross_entropy'):
            p = self.policy.class_probs
            p = p[0:T_squeezed, :]
            y_tiled = tf.tile(tf.reshape(self.y, [1, num_classes]), [tf.shape(p)[0], 1])
            self.cross_entropy = -tf.reduce_sum(y_tiled * tf.log(1e-10 + p), 1)
            self.cross_entropy = tf.reduce_mean(self.cross_entropy)

            p = tf.gather(p, self.T - 1, axis=0)
            p = tf.squeeze(p)
            self.guess = tf.squeeze(tf.math.argmax(p))

        with tf.variable_scope('loss'):
            loss = loss_clip + self.cross_entropy

        gradients = tf.gradients(loss, self.pi_trainable)
        # gradients[0] = self._flat_gradients(gradients[0])
        self.gradients = gradients

        # Get gradients and variables
        self.gradient_holder = []
        for j, var in enumerate(self.pi_trainable):
            self.gradient_holder.append(tf.placeholder(tf.float32, name = 'grads' + str(j)))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
        self.train_op = optimizer.apply_gradients(zip(self.gradient_holder, self.pi_trainable))

        self.actProbsOut = self.policy.act_probs[:, 1]

    def update(self, gradient_buffer):
        feed = dict(zip(self.gradient_holder, gradient_buffer))
        tf.get_default_session().run(self.train_op , feed_dict = feed)

    def getActProbs(self, stocks):
        actProbsOut = tf.get_default_session().run((self.actProbsOut), feed_dict={self.policy.stocks: stocks})
        return actProbsOut

    def get_vars(self):
        net_vars = tf.get_default_session().run(self.pi_trainable)
        return net_vars

    def _get_gaes(self, racc, c):
        gaes = np.ones(int(c)) * (-self.mu)
        gaes[-1] = gaes[-1] + racc # - np.log(0.5)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes

    def get_grads(self, stocks, y, T, T_slicer):
        ra = tf.get_default_session().run(self.reward_acc, feed_dict={self.policy.stocks: stocks,
                                                                                              self.old_policy.stocks: stocks,
                                                                                              self.y: y, self.T: T, self.T_slicer: T_slicer})
        discounted_rewards = self._get_gaes(ra, T)

        # center discounted reward signal
        discounted_rewards = (discounted_rewards-np.mean(discounted_rewards)) # / np.std(discounted_rewards)

        grad, guess, ce, r, r_acc, t_stop = tf.get_default_session().run((self.gradients, self.guess, self.cross_entropy, self.reward, self.reward_acc, self.time_stop),
                                                                          feed_dict={self.policy.stocks: stocks,
                                                                                     self.old_policy.stocks: stocks,
                                                                                     self.y: y, self.T: T, self.T_slicer: T_slicer,
                                                                                     self.should_be_gae: discounted_rewards})

        return grad, guess, ce, r, r_acc, t_stop

    def _run_oneMonth_policy(self, stocks):
        act_probs, class_probs = tf.get_default_session().run((self.policy.act_probs, self.policy.class_probs),
                                                              feed_dict={self.policy.stocks: stocks})

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

    def assign_policy_parameters(self):
        # assign policy parameter values to old policy parameters
        return tf.get_default_session().run(self.assign_ops)

    # def get_gaes(self, rewards, v_preds, v_preds_next):
    # ...

    # This is needed for tf.gather like operations.
    # def _flat_gradients(self, grads_or_idx_slices):
    #     '''Convert gradients if it's tf.IndexedSlices.
    #     When computing gradients for operation concerning `tf.gather`, the type of gradients
    #     '''
    #     # if type(grads_or_idx_slices) == tf.IndexedSlices:
    #     return tf.scatter_nd(
    #         tf.expand_dims(grads_or_idx_slices.indices, 1),
    #         grads_or_idx_slices.values,
    #         grads_or_idx_slices.dense_shape
    #     )
    #     # return grads_or_idx_slices
