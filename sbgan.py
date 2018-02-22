import tensorflow as tf
import pdb
from functools import partial


class SBGAN(object):
    def __init__(self,
            generator,
            discriminator,
            n_g=5,
            n_d=1,
            kernel="rbf"):
        self.generator = lambda scope: partial(generator, scope=scope) 
        self.discriminator = lambda scope: partial(discriminator, scope=scope)
        self.n_g = n_g
        self.n_d = n_d
        self.kernel = self._kernel(kernel)


    # TODO: adaptively change bandwidth (page 6, section 5)
    def _kernel(self,
            kernel):
        def _rbf(x1, x2):
            h = 1
            
            def _to_tf_vec(x):
                x = [tf.reshape(_x, [-1]) for _x in x]
                return tf.concat(x, 0)

            if type(x1).__name__ == 'list':
                x1 = _to_tf_vec(x1)
            if type(x2).__name__ == 'list':
                x2 = _to_tf_vec(x2)

            return tf.exp(-tf.reduce_sum((x1-x2)*(x1-x2))/h)
            
        if kernel == "rbf":
            return _rbf


    def _stein_phi_star(self,
            theta,
            particles,
            posterior):
        num_param = len(theta)
        phi_star = [0 for _ in range(num_param)]

        for i, particle in enumerate(particles):
            grad_log_post = tf.gradients(posterior[i], particle)
            grad_kernel = tf.gradients(self.kernel(particle, theta), \
                    particle)

            for j in range(num_param):
                phi_star[j] += self.kernel(particle, theta)*grad_log_post[j] + \
                        grad_kernel[j]

        for j in range(num_param):
            phi_star[j] /= len(particles)

        return phi_star


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

        return x, z, iterator


    def _prior(self,
            params_group,
            config):
        prior_loss = 0
        for param in params_group:
            param /= config.prior_std
            prior_loss -= tf.reduce_mean(tf.multiply(param, param))

        return prior_loss


    def train(self,
            sess,
            real_data,
            config,
            g_scope="generator",
            d_scope="discriminator",
            hooks=None):
        """
        config: should have the following properties
                x_batch_size
                z_batch_size
                z_dims
                z_std
                num_epochs
                num_svgd_iterations
                step_size 
                prior_std
        """
        def _get_var(scope):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                scope=scope)

        def _flatten(main_list):
            return [item for sub_list in main_list for item in sub_list]

        x, z, iterator = self._data_handler(config, real_data)
        eps = tf.placeholder(dtype=tf.float32)
        
        # network initialisation
        # TODO: initialize points from the prior (page 6, section 5)
        generators = [self.generator(g_scope+"_%d"%i) for i in range(self.n_g)]
        discriminators = [self.discriminator(d_scope+"_%d"%i) for i in range(self.n_d)]

        # posteriors
        post_g = [0. for _ in range(self.n_g)]
        for i in range(self.n_g):
            for j in range(self.n_d):
                post_g[i] += tf.reduce_mean(\
                        tf.log(discriminators[j](generators[i](z[0][i]))))
            post_g[i] /= self.n_d

        post_d = [0. for _ in range(self.n_d)]
        for i in range(self.n_d):
            post_d[i] += tf.reduce_mean(tf.log(discriminators[i](x[i])))
            for j in range(self.n_g):
                post_d[i] += tf.reduce_mean(\
                        tf.log(1.-discriminators[i](generators[j](z[1][j]))))
            post_d[i] /= self.n_g

        var_g = [_get_var(g_scope+"_%d"%i) for i in range(self.n_g)]
        var_d = [_get_var(d_scope+"_%d"%i) for i in range(self.n_d)]

        # priors
        prior_g = [self._prior(_var_g, config) for _var_g in var_g]
        prior_d = [self._prior(_var_d, config) for _var_d in var_d] 

        for i in range(self.n_g): post_g[i] += prior_g[i]
        for i in range(self.n_d): post_d[i] += prior_d[i]

        # train steps
        # TODO: annealing
        g_phi_star = [self._stein_phi_star(var_g[i], var_g, post_g) \
                for i in range(self.n_g)]
        g_train_steps = [[tf.assign(_var_g, _var_g+eps*g_phi_star[j][i]) for i, _var_g \
                in enumerate(var_g[j])] for j in range(self.n_g)]
        g_train_steps = _flatten(g_train_steps) 

        d_phi_star = [self._stein_phi_star(var_d[i], var_d, post_d) \
                for i in range(self.n_d)]
        d_train_steps = [[tf.assign(_var_d, _var_d+eps*d_phi_star[j][i]) for i, _var_d \
                in enumerate(var_d[j])] for j in range(self.n_d)]
        d_train_steps = _flatten(d_train_steps) 


        init = tf.global_variables_initializer()
        sess.run(init)

        # run
        for epoch in range(config.num_epochs): 
            print epoch
            sess.run(iterator.initializer)            

            while True:
                try:
                    sess.run(g_train_steps, {eps: config.step_size})
                    sess.run(d_train_steps, {eps: config.step_size})

                except tf.errors.OutOfRangeError:
                    break


