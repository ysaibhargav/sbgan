from __future__ import print_function

import tensorflow as tf
import numpy as np
import pdb
from functools import partial
from itertools import combinations
import pdb
import logging
import inspect


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


    def _to_tf_vec(self,
            x):
        """ flattens every tensor in x and concats """
        x = [tf.reshape(_x, [-1]) for _x in x]
        return tf.concat(x, 0)


    def _bandwidth(self,
            particles):
        """ adaptively computes the bandwidth to make kernels sum to ~1 """
        if len(particles) == 1: return tf.constant(1.)
        distances = []
        _particles = [self._to_tf_vec(x) for x in particles]
        for x1, x2 in combinations(_particles, 2):
            distances.append(tf.sqrt(tf.reduce_sum((x1-x2)*(x1-x2))))

        distances = tf.convert_to_tensor(distances)
        m = tf.shape(distances)[0]//2
        median = tf.nn.top_k(distances, m).values[tf.maximum(m-1, 0)]

        return median**2/np.log(len(particles))


    def _kernel(self,
            kernel):
        def _rbf(x1, x2):
            if type(x1).__name__ == 'list':
                x1 = self._to_tf_vec(x1)
            if type(x2).__name__ == 'list':
                x2 = self._to_tf_vec(x2)

            return tf.exp(-tf.reduce_sum((x1-x2)*(x1-x2))/self.bandwidth)
            
        self.bandwidth = tf.placeholder(dtype=tf.float32, shape=())
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
            kernel_dist = self.kernel(particle, theta)

            for j in range(num_param):
                phi_star[j] += kernel_dist*grad_log_post[j] + grad_kernel[j]

        for j in range(num_param):
            phi_star[j] /= len(particles)
            #phi_star[j] = tf.clip_by_norm(phi_star[j], 10)

        return phi_star 


    def _prior(self,
            params_group, config):
        if config.prior == 'xavier':
            return 0.

        elif config.prior == 'normal':
            prior_loss = 0
            for param in params_group:
                # TODO: why is this reduce_mean in BGAN?
                prior_loss -= tf.reduce_mean(tf.multiply(param, param))
            prior_loss /= config.prior_std ** 2

            return prior_loss / 2

    def _unsupervised_posterior(self, generators, discriminators, data, config, N):

        with tf.name_scope('unsupervised_posterior/gen/'):
            post_g = [0. for _ in range(self.n_g)]
            g_labels_real = tf.constant(1., shape = [config.z_batch_size, 1])
            for i in range(self.n_g):
                for j in range(self.n_d):
                    post_g[i] -= tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=g_labels_real,
                            logits=discriminators[j](generators[i](data.z[0][i]))
                        )
                    )
                post_g[i] *= N
        
        with tf.name_scope('unsupervised_posterior/disc/'):
            post_d = [0. for _ in range(self.n_d)]
            d_labels_real = tf.constant(1., shape=(config.x_batch_size, 1))
            d_labels_fake = tf.constant(0., shape=(config.z_batch_size, 1))
            for i in range(self.n_d):
                post_d[i] -= tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=d_labels_real,
                        logits=discriminators[i](data.x[i])))
                for j in range(self.n_g):
                    post_d[i] -= tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=d_labels_fake,
                            logits=discriminators[i](generators[j](data.z[1][j]))
                        )
                    )
                post_d[i] *= N
        
        return post_g, post_d

    def _semisupervised_posterior(self, generators, discriminators, data, config, N):
        num_classes = data.n_classes
        with tf.name_scope('semisupervised_posterior/gen/'):
            post_g = [0. for _ in range(self.n_g)]
            g_labels_real = tf.constant([[0.] + [1. / num_classes] * num_classes] * 
                    config.z_batch_size)
            for i in range(self.n_g):
                for j in range(self.n_d):

                    logits = discriminators[j](generators[i](data.z[0][i]))

                    prob_except_fake = tf.reduce_logsumexp(logits[:, 1:], axis = 1)
                    prob = tf.reduce_logsumexp(logits, axis = 1)
                    post_g[i] += tf.reduce_mean(prob_except_fake - prob)

                post_g[i] *= N
        
        with tf.name_scope('semisupervised_posterior/disc/'):
            post_d = [0. for _ in range(self.n_d)]
            d_labels_fake = tf.constant([[1.] + [0.] * num_classes] * config.z_batch_size)
            d_labels_real = tf.constant([[0.] + [1. / num_classes] * num_classes] * 
                    config.x_batch_size)
            d_labels_classes = tf.concat(values=[tf.constant(0., shape=[config.n_supervised, 
                1]), data.ys], axis=1)
            for i in range(self.n_d):
                'real samples'

                logits = discriminators[i](data.x[i])
                prob_except_fake = tf.reduce_logsumexp(logits[:, 1:], axis = 1)
                prob = tf.reduce_logsumexp(logits, axis = 1)
                post_d[i] += tf.reduce_mean(prob_except_fake - prob)

                """
                'semi supervised'
                post_d[i] -= tf.reduce_sum(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=d_labels_classes,
                        logits=discriminators[i](data.xs)
                    )
                )
                """
                
                'generated samples'
                for j in range(self.n_g):
                    post_d[i] -= tf.reduce_sum(
                        tf.nn.softmax_cross_entropy_with_logits(
                            labels=d_labels_fake,
                            logits=discriminators[i](generators[j](data.z[1][j]))
                        )
                    )
                post_d[i] *= N

        return post_g, post_d

    
    def train(self,
            sess,
            config,
            data,
            g_scope="generator",
            d_scope="discriminator",
            summary=False,
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
        N = 1#len(data._data['train']['x'])

        def _get_var(scope):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                scope=scope)

        def _flatten(main_list):
            return [item for sub_list in main_list for item in sub_list]

        #data, iterator = self._data_handler(config, real_data)
        eps = tf.placeholder(dtype=tf.float32)
        
        # network initialisation
        # initialize points from the prior (page 6, section 5)
        generators = [self.generator(g_scope+"_%d_"%i) for i in range(self.n_g)]
        discriminators = [self.discriminator(d_scope+"_%d_"%i) for i in range(self.n_d)]

        if config.exp == 'unsupervised':
            post_g, post_d = self._unsupervised_posterior(generators, discriminators, data, 
                    config, N)
        elif config.exp == 'semisupervised':
            post_g, post_d = self._semisupervised_posterior(generators, discriminators, data, 
                    config, N)
        
        var_g = [_get_var(g_scope+"_%d_"%i) for i in range(self.n_g)]
        var_d = [_get_var(d_scope+"_%d_"%i) for i in range(self.n_d)]

        # priors
        prior_g = [self._prior(_var_g, config) for _var_g in var_g]
        prior_d = [self._prior(_var_d, config) for _var_d in var_d] 

        for i in range(self.n_g): post_g[i] += prior_g[i]
        for i in range(self.n_d): post_d[i] += prior_d[i]

        g_bandwidth = self._bandwidth(var_g)
        d_bandwidth = self._bandwidth(var_d)

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

        if summary:
            for i in range(self.n_g):
                tf.summary.scalar('prior_g_%i'%i, prior_g[i])
                tf.summary.scalar('likelihood_g_%i'%i, post_g[i]-prior_g[i])
                tf.summary.scalar('post_g_%i'%i, post_g[i])

                for j, _var_g in enumerate(var_g[i]):
                    tf.summary.histogram('generator_%i/phi_star_%s'%(i, _var_g.name), 
                            g_phi_star[i][j])

            for i in range(self.n_d):
                tf.summary.scalar('prior_d_%i'%i, prior_d[i])
                tf.summary.scalar('likelihood_d_%i'%i, post_d[i]-prior_d[i])
                tf.summary.scalar('post_d_%i'%i, post_d[i])
                
                for j, _var_d in enumerate(var_d[i]):
                    tf.summary.histogram('discriminator_%i/phi_star_%s'%(i, _var_d.name), 
                            d_phi_star[i][j])

            tf.summary.scalar('g_bandwidth', g_bandwidth)

            tf.summary.scalar('d_bandwidth', d_bandwidth)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)

            merged_summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(config.summary_savedir, 
                    graph=tf.get_default_graph())

        init = tf.global_variables_initializer()
        sess.run(init)

        update_iter = 0
        # run
        for epoch in range(config.num_epochs): 
            print(epoch)
            sess.run(data.unsupervised_iterator.initializer)            

            # TODO: multiple opt steps 
            while True:
                if config.exp == 'semisupervised':
                    sess.run(data.supervised_iterator.initializer)
                try:
                    _g_bandwidth = sess.run(g_bandwidth)
                    sess.run(g_train_steps, {\
                            eps: config.step_size, 
                            self.bandwidth: _g_bandwidth})

                    _d_bandwidth = sess.run(d_bandwidth)
                    if summary and update_iter % config.summary_n == 0:
                        _summary, _ = sess.run([merged_summary_op, d_train_steps], {\
                                eps: config.step_size,
                                self.bandwidth: _d_bandwidth})
                        summary_writer.add_summary(_summary, update_iter)
                        summary_writer.flush()
                    else:
                        sess.run(d_train_steps, {\
                                eps: config.step_size,
                                self.bandwidth: _d_bandwidth})

                    update_iter += 1

                except tf.errors.OutOfRangeError:
                    break

            if hooks != None:

                for hook in hooks:
                    if epoch % hook.frequency == 0:
                        out = sess.run([generator(data.z[0][i]) for i, generator in \
                                enumerate(generators)])
                        if config.exp == 'semisupervised':
                            self.test(sess, data, d_scope='discriminator')

                        if hook.is_joint:
                            hook.function(**{"g_z": out, 
                                "real_data": data._data['train']['x'],
                                "epoch": "%d"%(epoch)})
                        else:
                            for i, _out in enumerate(out):
                                hook.function(**{"g_z": _out,
                                    "epoch": "%d_%d"%(epoch, i)})

    def test(self, sess, data, d_scope='discriminator'):
        '''
        testing graph and loop
        '''
        discriminators = []
        for i in range(self.n_d):
            discriminator = self.discriminator(d_scope+"_%d_"%i)
            if 'train' in inspect.getfullargspec(discriminator).args:
                discriminator = partial(discriminator, train=False)
            discriminators.append(discriminator)
            
        p = 0.
        for i in range(self.n_d):
            '''
            compute predictions from 1 discriminator
            '''
            p += tf.nn.softmax(
                logits = discriminators[i](data.x_test)[:, 1:],
                dim=-1
            )
        p /= self.n_d
        predictions = tf.argmax(p, axis = 1)
        actual = tf.argmax(data.y_test, axis = 1)
        correct = tf.reduce_sum(tf.cast(tf.equal(predictions, actual), dtype=tf.float32))
        total_samples = 0
        total_correct = 0
        sess.run(data.test_iterator.initializer)
        while True:
            try:
                x, num_correct = sess.run([data.x_test, correct])
                total_samples += x.shape[0]
                total_correct +=  num_correct

            except tf.errors.OutOfRangeError:
                break
        #pdb.set_trace()
        logger = logging.getLogger()
        logger.info('Test Accuracy: %.2f' % (100. * total_correct / total_samples))
