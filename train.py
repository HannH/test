#coding:utf-8
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import vlib.plot as plot
import vlib.save_images as save_img
from tensorflow.examples.tutorials.mnist import input_data  # as mnist_data
from vlib.layers import *
from vlib.load_data import *

mnist = input_data.read_data_sets('data/', one_hot=True)
# temp = 0.89
class Train(object):
    def __init__(self, sess, args):
        #sess=tf.Session()
        self.sess = sess
        self.img_size = 28   # the size of image
        self.trainable = True
        self.batch_size = 50  # must be even number
        self.lr = 2e-4
        self.mm = 0.5      # momentum term for adam
        self.z_dim = 128   # the dimension of noise z
        self.EPOCH = 50    # the number of max epoch
        self.LAMBDA = 0.1  # parameter of WGAN-GP
        self.model = args.model  # 'DCGAN' or 'WGAN'
        self.dim = 1       # RGB is different with gray pic
        self.num_class = 11
        self.load_model = args.load_model
        self.label_num=args.label_num
        if self.model=='ct-gan':self.build_model_ctgan()
        else:self.build_model()  # initializer

    def build_model(self):
        # build  placeholders
        self.x=tf.placeholder(tf.float32,shape=[self.batch_size,self.img_size*self.img_size*self.dim],name='real_img')
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim], name='noise')
        self.label = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_class - 1], name='label')
        self.flag = tf.placeholder(tf.float32, shape=[], name='flag')
        self.flag2 = tf.placeholder(tf.float32, shape=[], name='flag2')

        # define the network
        self.G_img = self.generator('gen', self.z, reuse=False)
        d_logits_r, layer_out_r = self.discriminator('dis', self.x, reuse=False)
        d_logits_f, layer_out_f = self.discriminator('dis', self.G_img, reuse=True)

        d_regular = tf.add_n(tf.get_collection('regularizer', 'dis'), 'loss')  # D regular loss
        # caculate the unsupervised loss
        un_label_r = tf.concat([tf.ones_like(self.label), tf.zeros(shape=(self.batch_size, 1))], axis=1)
        un_label_f = tf.concat([tf.zeros_like(self.label), tf.ones(shape=(self.batch_size, 1))], axis=1)
        logits_r, logits_f = tf.nn.softmax(d_logits_r), tf.nn.softmax(d_logits_f)
        # d_loss_r = - tf.reduce_mean(0.75*tf.log(logits_r[:, :-1]))
        # d_loss_f = - tf.reduce_mean(0.75*tf.log(1-logits_f[:, -1]))
        # d_loss_r = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(d_logits_r[:,:-1]),logits_r[:, :-1]))
        # d_loss_f = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.zeros_like(d_logits_f[:, -1]),logits_f[:, -1]))
        d_loss_r = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_r[:, -1]),
                                                       logits=d_logits_r[:, -1]))
        d_loss_f = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_f[:, -1])*0.9, logits=d_logits_f[:, -1]))
        d_loss_f1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_f[:, -1]),
                                                    logits=d_logits_f[:, -1]))
        # d_loss_r = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=un_label_r*0.9, logits=d_logits_r))
        # d_loss_f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=un_label_f*0.9, logits=d_logits_f))
        # feature match
        f_match = []
        for i in range(4):
            f_match += [tf.reduce_mean(tf.multiply(layer_out_f[i]-layer_out_r[i], layer_out_f[i]-layer_out_r[i]))]

        # caculate the supervised loss
        s_label = tf.concat([self.label, tf.zeros(shape=(self.batch_size,1))], axis=1)
        s_l_r = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=s_label, logits=d_logits_r))
        self.d_l_1, self.d_l_2 = d_loss_r + d_loss_f, s_l_r
        self.d_loss = d_loss_r + d_loss_f + s_l_r*self.flag+d_regular
        self.g_loss = d_loss_f1+0.1*tf.reduce_mean(f_match,0)

        all_vars = tf.global_variables()
        g_vars = [v for v in all_vars if 'gen' in v.name]
        d_vars = [v for v in all_vars if 'dis' in v.name]
        for v in all_vars:
            print(v)
        if self.model == 'DCGAN':
            self.opt_d = tf.train.AdamOptimizer(self.lr, beta1=self.mm).minimize(self.d_loss, var_list=d_vars)
            self.opt_g = tf.train.AdamOptimizer(self.lr, beta1=self.mm).minimize(self.g_loss, var_list=g_vars)
        elif self.model == 'WGAN_GP':
            self.opt_d = tf.train.AdamOptimizer(1e-5, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=d_vars)
            self.opt_g = tf.train.AdamOptimizer(1e-5, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=g_vars)
        elif self.model == 'nogan':
            self.opt_d = tf.train.AdamOptimizer(1e-5, beta1=0.5, beta2=0.9).minimize(s_l_r, var_list=d_vars)
            self.opt_g = tf.constant(0)
        else:
            print ('model can only be "DCGAN","WGAN_GP" !')
            return
        # test
        test_logits, _ = self.discriminator('dis', self.x, reuse=True)
        test_logits = tf.nn.softmax(test_logits)
        temp = tf.reshape(test_logits[:, -1],shape=[self.batch_size, 1])
        for i in range(10):
            temp = tf.concat([temp, tf.reshape(test_logits[:, -1],shape=[self.batch_size, 1])], axis=1)
        test_logits -= temp
        self.prediction = tf.nn.in_top_k(test_logits, tf.argmax(s_label, axis=1), 1)

        self.saver = tf.train.Saver()
        if not self.load_model:
            init = tf.global_variables_initializer()
            self.sess.run(init)
        elif self.load_model:
            self.saver.restore(self.sess, os.getcwd()+'/model_saved/model.ckpt')
            print('model load done')
        self.sess.graph.finalize()

    def build_model_ctgan(self):
        if self.model!='ct-gan':
            print('model can only be “ct-gan” !')
            return
        # build  placeholders
        self.x=tf.placeholder(tf.float32,shape=[self.batch_size,self.img_size*self.img_size*self.dim],name='real_img')
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim], name='noise')
        self.label = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_class - 1], name='label')
        self.flag = tf.placeholder(tf.float32, shape=[], name='flag')

        # define the network
        self.G_img = self.generator('gen', self.z, reuse=False)
        d_logits_r1, d_logits_r11,d_logits_r12 = self.discriminator_with_dropout('dis', self.x, reuse=False)
        d_logits_r2, d_logits_r21,_ = self.discriminator_with_dropout('dis', self.x, reuse=True)
        d_logits_f, _ ,d_logits_f2 = self.discriminator_with_dropout('dis', self.G_img, reuse=True)

        d_regular = tf.add_n(tf.get_collection('regularizer', 'dis'), 'loss')  # D regular loss
        # caculate the unsupervised loss
        logits_r, logits_f = tf.nn.softmax(d_logits_r1), tf.nn.softmax(d_logits_f)
        d_loss_r = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_r[:, -1]), logits=d_logits_r1[:, -1]))
        d_loss_f = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_f[:, -1]), logits=d_logits_f[:, -1]))
        loss_ct=tf.square(d_logits_r1-d_logits_r2)
        loss_ct_=0.1*tf.reduce_mean(tf.square(d_logits_r11-d_logits_r21))
        CT=loss_ct+loss_ct_
        # feature match
        # caculate the supervised loss
        s_label = tf.concat([self.label, tf.zeros(shape=(self.batch_size,1))], axis=1)
        s_l_r = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=s_label, logits=d_logits_r1))
        self.d_l_1, self.d_l_2 = d_loss_r + d_loss_f, s_l_r
        self.d_loss = d_loss_r + d_loss_f + s_l_r*self.flag +0.1*tf.reduce_mean(CT)
        self.g_loss = tf.square(tf.reduce_mean(d_logits_f2,0)-tf.reduce_mean(d_logits_r12,0))
        all_vars = tf.global_variables()
        for v in all_vars:
            print(v)
        all_vars = tf.global_variables()
        g_vars = [v for v in all_vars if 'gen' in v.name]
        d_vars = [v for v in all_vars if 'dis' in v.name]
        self.opt_d = tf.train.AdamOptimizer(self.lr).minimize(self.d_loss, var_list=d_vars)
        self.opt_g = tf.train.AdamOptimizer(self.lr).minimize(self.g_loss, var_list=g_vars)
        test_logits, _,_ = self.discriminator_with_dropout('dis', self.x, reuse=True,keep_prob=1.0)
        test_logits = tf.nn.softmax(test_logits)
        # temp = tf.reshape(test_logits[:, -1],shape=[self.batch_size, 1])
        # for i in range(10):
        #     temp = tf.concat([temp, tf.reshape(test_logits[:, -1],shape=[self.batch_size, 1])], axis=1)
        self.prediction = tf.nn.in_top_k(test_logits, tf.argmax(s_label, axis=1), 1)

        self.saver = tf.train.Saver()
        if not self.load_model:
            init = tf.global_variables_initializer()
            self.sess.run(init)
        elif self.load_model:
            self.saver.restore(self.sess, os.getcwd()+'/model_saved/model.ckpt')
            print('model load done')
        self.sess.graph.finalize()
    def train(self):
        if not os.path.exists('model_saved'):
            os.mkdir('model_saved')
        if not os.path.exists('gen_picture'):
            os.mkdir('gen_picture')
        noise = np.random.normal(-1, 1, [self.batch_size, 128])
        temp = 0.80
        print('training')
        indexrange = list(range(self.batch_size * self.label_num))
        g_opt = [self.opt_g, self.g_loss]
        d_opt = [self.opt_d, self.d_loss, self.d_l_1, self.d_l_2]
        for epoch in range(self.EPOCH):
            # iters = int(156191//self.batch_size)
            iters = 50000//self.batch_size
            true_batchx, true_batchl = mnist.train.next_batch(self.batch_size * self.label_num)
            for idx in range(iters):
                flag = 1 if idx < self.label_num and epoch==0 else 0 # set we use 2*batch_size=200 train data labeled.
                batchx, batchl = mnist.train.next_batch(self.batch_size)
                if np.random.rand()>1:
                    feed = {self.x:batchx, self.z:noise, self.label:batchl, self.flag:0}
                else :
                    np.random.shuffle(indexrange)
                    feed = {self.x:true_batchx[indexrange[:self.batch_size]],
                            self.z:noise, self.label:true_batchl[indexrange[:self.batch_size]], self.flag:1}
                # update the Discrimater k times
                _, loss_d, d1,d2 = self.sess.run(d_opt, feed_dict=feed)
                # update the Generator one time
                _, loss_g = self.sess.run(g_opt, feed_dict=feed)
                # print(("[%3f][epoch:%2d/%2d][iter:%4d/%4d],loss_d:%5f,loss_g:%4f, d1:%4f, d2:%4f"%
                #        (time.time()-start_t, epoch, self.EPOCH,idx,iters, loss_d, loss_g,d1,d2)), 'flag:',flag)
                plot.plot('d_loss', loss_d)
                plot.plot('g_loss', loss_g)
                if ((idx+1) % 100) == 0:  # flush plot picture per 1000 iters
                    plot.flush()
                plot.tick()
                if (idx+1)%500==0:
                    print ('images saving............')
                    img = self.sess.run(self.G_img, feed_dict=feed)
                    save_img.save_images(img, os.getcwd() +'/gen_picture/' +'sample{}_{}.jpg' \
                                         .format(epoch, (idx + 1) // 500))
                    print('images save done')
            test_acc = self.test()
            plot.plot('test acc', test_acc)
            plot.flush()
            plot.tick()
            print('test acc:{}'.format(test_acc), 'temp:%3f' % (temp))
            if test_acc > temp:
                print ('model saving..............')
                path = os.getcwd() + '/model_saved'
                save_path = os.path.join(path, "model.ckpt")
                self.saver.save(self.sess, save_path=save_path)
                print ('model saved...............')
                temp = test_acc

# output = conv2d('Z_cona{}'.format(i), output, 3, 64, stride=1, padding='SAME')

    def generator(self,name, noise, reuse):
        with tf.variable_scope(name,reuse=reuse):
            l = self.batch_size
            output = fc('g_dc', noise, 2*2*64)
            output = tf.reshape(output, [-1, 2, 2, 64])
            output = tf.nn.relu(self.bn('g_bn1',output))
            output = deconv2d('g_dcon1',output,5,outshape=[l, 4, 4, 64*4])
            output = tf.nn.relu(self.bn('g_bn2',output))

            output = deconv2d('g_dcon2', output, 5, outshape=[l, 8, 8, 64 * 2])
            output = tf.nn.relu(self.bn('g_bn3', output))

            output = deconv2d('g_dcon3', output, 5, outshape=[l, 16, 16,64 * 1])
            output = tf.nn.relu(self.bn('g_bn4', output))

            output = deconv2d('g_dcon4', output, 5, outshape=[l, 32, 32, self.dim])
            output=tf.sigmoid(output)
            output = tf.image.resize_images(output, (28, 28))
            # output = tf.nn.relu(self.bn('g_bn4', output))
            return tf.nn.tanh(output)

    def discriminator(self, name, inputs, reuse):
        l = tf.shape(inputs)[0]
        inputs = tf.reshape(inputs, (l,self.img_size,self.img_size,self.dim))
        with tf.variable_scope(name,reuse=reuse):
            out = []
            output = conv2d('d_con1',inputs,5, 64, stride=2, padding='SAME') #14*14
            output1 = lrelu(self.bn('d_bn1',output))
            out.append(output1)
            # output1 = tf.contrib.keras.layers.GaussianNoise
            output = conv2d('d_con2', output1, 3, 64*2, stride=2, padding='SAME')#7*7
            output2 = lrelu(self.bn('d_bn2', output))
            out.append(output2)
            output = conv2d('d_con3', output2, 3, 64*4, stride=1, padding='VALID')#5*5
            output3 = lrelu(self.bn('d_bn3', output))
            out.append(output3)
            output = conv2d('d_con4', output3, 3, 64*4, stride=2, padding='VALID')#2*2
            output4 = lrelu(self.bn('d_bn4', output))
            out.append(output4)
            output = tf.reshape(output4, [l, 2*2*64*4])# 2*2*64*4
            output = fc('d_fc', output, self.num_class)
            # output = tf.nn.softmax(output)
            return output, out

    def discriminator_with_dropout(self, name, inputs, reuse,keep_prob=0.8):
        l = tf.shape(inputs)[0]
        inputs = tf.reshape(inputs, (l,self.img_size,self.img_size,self.dim))
        with tf.variable_scope(name,reuse=reuse):
            output = conv2d('d_con1',inputs,5, 64, stride=2, padding='SAME') #14*14
            output1 = lrelu(self.bn('d_bn1',output))
            output1 = tf.nn.dropout(output1,keep_prob)
            # output1 = tf.contrib.keras.layers.GaussianNoise
            output = conv2d('d_con2', output1, 3, 64*2, stride=2, padding='SAME')#7*7
            output2 = lrelu(self.bn('d_bn2', output))
            output2 = tf.nn.dropout(output2,keep_prob)
            output = conv2d('d_con3', output2, 3, 64*4, stride=1, padding='VALID')#5*5
            output3 = lrelu(self.bn('d_bn3', output))
            output3 = tf.nn.dropout(output3,keep_prob)
            output = conv2d('d_con4', output3, 3, 64*4, stride=2, padding='VALID')#2*2
            output4 = lrelu(self.bn('d_bn4', output))
            output4 = tf.nn.dropout(output4,keep_prob)
            output5 = tf.reshape(output4, [l, 2*2*64*4])# 2*2*64*4
            output6 =  tf.nn.dropout(output5,keep_prob)
            output = fc('d_fc', output6, self.num_class)
            # output = tf.nn.softmax(output)
            return output, output6, output5

    def bn(self, name, input):
        val = tf.contrib.layers.batch_norm(input, decay=0.9,
                                           updates_collections=None,
                                           epsilon=1e-5,
                                           scale=True,
                                           is_training=True,
                                           scope=name)
        return val

    # def get_loss(self, logits, layer_out):
    def test(self):
        count = 0.
        print('testing................')
        for i in range(10000//self.batch_size):
            testx, textl = mnist.test.next_batch(self.batch_size)
            prediction = self.sess.run(self.prediction, feed_dict={self.x:testx, self.label:textl})
            count += np.sum(prediction)
        return count/10000.
