import tensorflow as tf


class SBGAN(object):
    def __init__(self,
            generator,
            discriminator,
            n_g=5,
            n_d=1
            kernel="rbf"):
        # TODO: property for tf scope argument
        self.generator = generator
        self.discriminator = discriminator 
        self.n_g = n_g
        self.n_d = n_d
        self.kernel = self._kernel(kernel)


    def _kernel(self,
            kernel):
        pass


    def _stein_phi_star(self,
            theta,
            particles,
            posterior,
            config):
        pass


    def _data_handler(self,
            config,
            real_data):
        round_sz = config.x_batch_size*(real_data.shape[0]\
                    //config.x_batch_size)
        real_data = real_data[:round_sz]

        dataset = tf.data.Dataset.from_tensor_slices((real_data))
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(config.x_batch_size)
        iterator = dataset.make_initializable_iterator()

        x = [iterator.get_next() for _ in range(self.n_d)]
        z = tf.random_normal([2, self.n_g, config.z_batch_size, \
                config.z_dims], stddev = config.z_std)

        return x, z


    def _prior(self,
            params):
        pass


    def train(self,
            sess,
            g_optimizer,
            d_optimizer,
            real_data,
            config,
            g_scope = "generator",
            d_scope = "discriminator",
            hooks = None):
        """
        config: should have the following properties
                x_batch_size
                z_batch_size
                z_dims
                z_std
                num_epochs
                num_svgd_iterations
                learning_rate 
                prior_std
        """
        def _get_var(scope):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                scope=scope)

        def _flatten(main_list):
            return [item for sub_list in main_list for item in sub_list]

        x, z = self._data_handler(config, real_data)
        eps = tf.placeholder(dtype=tf.float32)
        
        # network initialisation
        generators = [self.generator(g_scope+"_%d"%i) for i in range(n_g)]
        discriminators = [self.discriminator(d_scope+"_%d"%i) for i in range(n_d)]

        var_g = [_get_var(g_scope+"_%d"%i) for i in range(n_g)]
        var_d = [_get_var(d_scope+"_%d"%i) for i in range(n_d)]

        # priors
        prior_g = [self._prior(_var_g) for _var_g in var_g]
        prior_d = [self._prior(_var_d) for _var_d in var_d] 

        # posteriors
        post_g = [None for _ in range(n_g)]
        for i in range(n_g):
            post_g[i] = 0 
            for j in range(n_d):
                post_g[i] += tf.reduce_mean(\
                        tf.log(discriminators[j](generators[i](z[0][i]))))
            post_g[i] /= n_d
            post_g[i] += prior_g[i]

        post_d = [None for _ in range(n_d)]
        for i in range(n_d):
            post_d[i] = 0 
            post_d[i] += tf.reduce_mean(tf.log(discriminators[i](x[i])))
            for j in range(n_g):
                post_d[i] += tf.reduce_mean(\
                        tf.log(1.-discriminators[i](generators[j](z[1][j]))))
            post_d[i] /= n_d
            post_d[i] += prior_d[i]

        # train steps
        # TODO: annealing
        g_phi_star = [self._svgd_phi_star(var_g[i], var_g, post_g[i], config)\
                for i in range(n_g)]
        g_train_steps = [[tf.assign(_var_g, _var_g+eps*g_phi_star[i]) for i, _var_g \
                in enumerate(var_g[j])] for j in range(ng)]
        g_train_steps = _flatten(g_train_steps) 

        d_phi_star = [self._svgd_phi_star(var_d[i], var_d, post_d[i], config)\
                for i in range(n_d)]
        d_train_steps = [[tf.assign(_var_d, _var_d+eps*d_phi_star[i]) for i, _var_d \
                in enumerate(var_d[j])] for j in range(nd)]
        d_train_steps = _flatten(d_train_steps) 


        init = tf.global_variables_initializer()
        sess.run(init)

        # run
        for epoch in range(config.num_epochs): 
            sess.run(iterator.initializer)            

            while True:
                try:
                    sess.run(g_train_steps, {eps: config.step_size})
                    sess.run(d_train_steps, {eps: config.step_size})

                except tf.errors.OutOfRangeError:
                    break




