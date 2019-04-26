import tensorflow as tf
from utils.train_op import *
from utils.vgg import *



class PPO_Net(object):

    def __init__(self, name, sess,dim_global_feature,dim_image_feature, a_dim, LR_A, LR_C, adaptive_dict, THRESHOLD,device = '/cpu:0'):
        with tf.device(device):
            with tf.name_scope(name):
                self.THRESHOLD = THRESHOLD
                self.adaptive_dict = adaptive_dict
                """
                    key:   KL_Min   KL_Max   KL_BETA
                """
                self.session = sess
                self.a_dim = a_dim
                self.s_image   = tf.placeholder(tf.float32, [None,WIDTH,HEIGHT,N_CHANNEL], name='state_image')
                self.s_feature = tf.placeholder(tf.float32, [None, dim_global_feature], name='state_feature')
                self.q_v = tf.placeholder(tf.float32, [None, 1], name='state_action_value')
                self.a = tf.placeholder(tf.int32, [None,1])
                self.adv_in_aloss = tf.placeholder(tf.float32, [None, 1], name='advantage')
                self.KL_BETA = tf.placeholder(tf.float32, None, 'KL_PUNISH_BETA')

                self.OPT_A = tf.train.AdamOptimizer(LR_A)
                self.OPT_C = tf.train.AdamOptimizer(LR_C)

                self.vgg_feature,self.vgg_params = model_vgg(input=self.s_image,model_path=VGG_PATH)
                # print("self.vgg_feature shape",self.vgg_feature.shape)
                # print("self.s_feature   shape",self.s_feature.shape)

                self.s = tf.concat([self.vgg_feature,self.s_feature],axis=1)
                # print("self.s shape:",self.s.shape)
                self._build_net(input=self.s,input_dim=dim_global_feature+dim_image_feature,name='PPO_Net')
                self._prepare_loss_and_train(name)
                # self.network_initial()

    def _distribution_net(self,input,input_dim,n_unit,name,trainable):
        # # continous action space
        with tf.variable_scope(name):
            # state ---> n_unit hidden_states
            w_encode = generate_fc_weight(shape=[input_dim, n_unit], name='w_encode', whe_train=trainable)
            b_encode = generate_fc_bias(shape=[n_unit], name='b_encode', whe_train=trainable)
            s_encode = tf.nn.relu6(tf.matmul(input, w_encode) + b_encode)

            w_critic = generate_fc_weight(shape=[n_unit, self.a_dim], name='w_critic', whe_train=trainable)
            b_critic = generate_fc_bias(shape=[self.a_dim], name='b_critic', whe_train=trainable)
            logits = tf.nn.relu6(tf.matmul(s_encode, w_critic) + b_critic)
            distribution = tf.nn.softmax(logits)
        params = [w_encode, b_encode, w_critic, b_critic]
        return distribution, params

    def _build_net(self, input,input_dim,name):
        with tf.variable_scope(name):

            with tf.variable_scope('Critic_Net'):
                # state ---> l_a
                self.w_c_encode = generate_fc_weight(shape=[input_dim, 2048], whe_train=True, name='w_c_encode')#
                self.b_c_encode = generate_fc_bias(shape=[2048], whe_train=True, name='b_c_encode')
                self.s_encode = tf.nn.relu6(tf.matmul(input, self.w_c_encode) + self.b_c_encode)

                self.w_critic = generate_fc_weight(shape=[2048, 1], whe_train=True, name='w_critic')
                self.b_critic = generate_fc_bias(shape=[1], whe_train=True, name='b_critic')
                self.value = tf.nn.relu6(tf.matmul(self.s_encode, self.w_critic) + self.b_critic)

            with tf.variable_scope('Actor_Net'):
                # state ---> new_distribution
                self.new_pi, self.new_params = self._distribution_net(input=input, input_dim=input_dim, n_unit=100,
                                                                      name='New_policy', trainable=True)
                # state ---> old_distribution   (old_pi is just used to stored the params,no need to be trained)
                self.old_pi, self.old_params = self._distribution_net(input=input, input_dim=input_dim, n_unit=100,
                                                                      name='Old_policy', trainable=False)
            with tf.variable_scope('sync_policy'):
                self.update_policy = [old.assign(new) for new, old in
                                      zip(self.new_params, self.old_params)]  # tf.assign(ref=new,value=old)


    def KL_divergence(self,p_current,p_target):
        X = tf.distributions.Categorical(probs = p_current)
        Y = tf.distributions.Categorical(probs = p_target )
        return tf.clip_by_value(tf.distributions.kl_divergence(X, Y), clip_value_min=0.0, clip_value_max=10)


    def _prepare_loss_and_train(self,name):
        with tf.variable_scope(name):

            with tf.variable_scope('Actor_loss'):
                # maximum  the "actor objective function":  [new_pi(s,a)/old_pi(s,a)]*[old_pi_based_advantage(s,a)] - BETA*KL_divergence
                self.new_pi_find_error = tf.reduce_sum(self.new_pi*tf.one_hot(self.a,self.a_dim),axis=2,keep_dims=False)
                self.old_pi_find_error = tf.reduce_sum(self.old_pi*tf.one_hot(self.a,self.a_dim),axis=2,keep_dims=False)
                self.pi_objective = ( self.new_pi_find_error/self.old_pi_find_error )*self.adv_in_aloss
                self.kl = self.KL_divergence(p_current=self.old_pi,p_target=self.new_pi)
                self.kl_objective = self.KL_BETA*self.kl
                # self.kl_mean is used for the computation of adapative of KL_BETA
                self.kl_mean = tf.reduce_mean(self.kl)
                self.a_loss = -tf.reduce_mean(self.pi_objective - self.kl_objective)

            with tf.variable_scope('Critic_loss'):
                self.advantage = self.q_v - self.value
                self.c_loss = tf.reduce_mean(tf.square(self.advantage))

            with tf.variable_scope('Actor_train'):
                self.a_train_op = self.OPT_A.minimize(self.a_loss)

            with tf.variable_scope('Critic_train'):
                self.c_train_op = self.OPT_C.minimize(self.c_loss)

    def choose_action(self,s_image,s_feature):
        prob_weights = self.session.run(self.new_pi,
                                        feed_dict={
                                                # self.s_image:   s_image[np.newaxis, :],
                                                # self.s_feature: s_feature[np.newaxis, :]
                                                    self.s_image: s_image[np.newaxis, :],
                                                    self.s_feature: s_feature[np.newaxis, :]
                                                   })
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action, prob_weights

    def get_value(self,s_image,s_feature):
        return self.session.run(self.value,{
                        self.s_image: s_image[np.newaxis, :],
                        self.s_feature: s_feature[np.newaxis, :]
                                    })[0,0]

    # def network_initial(self):
    #     self.session.run(tf.global_variables_initializer())

    def update_network(self, s_image,s_feature, a, q_value):
        self.session.run(self.update_policy)
        advantage = self.session.run(self.advantage, {self.s_image: s_image,
                                                      self.q_v: q_value,
                                                      self.s_feature:s_feature})

        # print("advantage:",advantage.shape)
        # print("s_feature shape:",s_feature.shape)
        # print("s_feature:",s_feature)

        # s_feature = np.reshape(s_feature,(-1,256))

        # update Actor_network
        for _ in range(A_ITER):
            _, kl = self.session.run([self.a_train_op, self.kl_mean],
                                      {self.s_image      : s_image,
                                       self.s_feature    : s_feature,
                                       self.adv_in_aloss : advantage,
                                       self.a            : a,
                                       self.KL_BETA      : self.adaptive_dict['KL_BETA']})
        # update Critic_network
        for _ in range(C_ITER):
            self.session.run(self.c_train_op, {self.s_image: s_image, self.s_feature:s_feature, self.q_v: q_value})
        # update adaptive params KL_BETA:
        #     if kl < kl_min ---> means the kl_divergance's power is too stronge , we need to dicrease the KL_BETA
        #     if kl > kl_max ---> means the kl_divergance's power is too weak    , we need to increase the KL_BETA
        if kl < self.adaptive_dict['KL_Min']:
            self.adaptive_dict['KL_BETA'] = self.adaptive_dict['KL_BETA'] / 2.0
        if kl > self.adaptive_dict['KL_Max']:
            self.adaptive_dict['KL_BETA'] = self.adaptive_dict['KL_BETA'] * 2.0
        self.adaptive_dict['KL_BETA'] = np.clip(self.adaptive_dict['KL_BETA'], 1e-4, 20)










