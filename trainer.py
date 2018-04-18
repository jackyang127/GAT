import time
import tensorflow as tf
import ray
import numpy as np

from models import GAT
from utils import process

#checkpt_file = 'pre_trained/cora/mod_cora.ckpt'
#
#dataset = 'cora'
#
## training params
#self.batch_size = 1
#nb_epochs = 100000
#self.patience = 100
#lr = 0.005  # learning rate
#l2_coef = 0.0005  # weight decay
#hid_units = [8] # numbers of hidden units per each attention head in each layer
#n_heads = [8, 1] # additional entry for the output layer
#residual = False
#nonlinearity = tf.nn.elu
#model = GAT

@ray.remote
class GAT_Run(object):
    def __init__(self):
        self.checkpt_file = 'pre_trained/cora/mod_cora.ckpt'

        dataset = 'cora'

        # training params
        self.batch_size = 1
        self.nb_epochs = 100000
        self.patience = 100
        lr = 0.005  # learning rate
        l2_coef = 0.0005  # weight decay
        hid_units = [8] # numbers of hidden units per each attention head in each layer
        n_heads = [8, 1] # additional entry for the output layer
        residual = False
        nonlinearity = tf.nn.elu
        model = GAT

        print('Starting timer')
        print('Dataset: ' + dataset)
        print('----- Opt. hyperparams -----')
        print('lr: ' + str(lr))
        print('l2_coef: ' + str(l2_coef))
        print('----- Archi. hyperparams -----')
        print('nb. layers: ' + str(len(hid_units)))
        print('nb. units per layer: ' + str(hid_units))
        print('nb. attention heads: ' + str(n_heads))
        print('residual: ' + str(residual))
        print('nonlinearity: ' + str(nonlinearity))
        print('model: ' + str(model))

        self.start = time.time()
        (self.adj, 
         self.features, 
         self.y_train, 
         self.y_val, 
         self.y_test, 
         self.train_mask, 
         self.val_mask, 
         self.test_mask) = process.load_data(dataset)
        self.features, self.spars = process.preprocess_features(self.features)

        self.nb_nodes = self.features.shape[0]
        self.ft_size = self.features.shape[1]
        self.nb_classes = self.y_train.shape[1]

        self.adj = self.adj.todense()

        self.features = self.features[np.newaxis]
        self.adj = self.adj[np.newaxis]
        self.y_train = self.y_train[np.newaxis]
        self.y_val = self.y_val[np.newaxis]
        self.y_test = self.y_test[np.newaxis]
        self.train_mask = self.train_mask[np.newaxis]
        self.val_mask = self.val_mask[np.newaxis]
        self.test_mask = self.test_mask[np.newaxis]

        self.biases = process.adj_to_bias(self.adj, [self.nb_nodes], nhood=1)

        with tf.name_scope('input'):
            self.ftr_in = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.nb_nodes, self.ft_size), name="ftr_in")
            self.bias_in = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.nb_nodes, self.nb_nodes), name="bias_in")
            self.lbl_in = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.nb_nodes, self.nb_classes), name="lbl_in")
            self.msk_in = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.nb_nodes), name="msk_in")
            self.attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name="attn_drop")
            self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name="ffd_drop")
            self.is_train = tf.placeholder(dtype=tf.bool, shape=(), name="is_train")

        self.logits = model.inference(self.ftr_in, self.nb_classes, self.nb_nodes, self.is_train,
                                    self.attn_drop, self.ffd_drop,
                                    bias_mat=self.bias_in,
                                    hid_units=hid_units, n_heads=n_heads,
                                    residual=residual, activation=nonlinearity)
        self.log_resh = tf.reshape(self.logits, [-1, self.nb_classes])
        self.lab_resh = tf.reshape(self.lbl_in, [-1, self.nb_classes])
        self.msk_resh = tf.reshape(self.msk_in, [-1])
        self.loss = model.masked_softmax_cross_entropy(self.log_resh, self.lab_resh, self.msk_resh)
        self.accuracy = model.masked_accuracy(self.log_resh, self.lab_resh, self.msk_resh)

        self.train_op = model.training(self.loss, lr, l2_coef)

        self.saver = tf.train.Saver()

        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        self.vlss_mn = np.inf
        self.vacc_mx = 0.0
        self.curr_step = 0
        self.sess = tf.Session()
        self.sess.run(self.init_op)

    def train(self):
        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0
        for epoch in range(self.nb_epochs):
            tr_step = 0
            tr_size = self.features.shape[0]

            while tr_step * self.batch_size < tr_size:
                _, loss_value_tr, acc_tr = self.sess.run([self.train_op, self.loss, self.accuracy],
                    feed_dict={
                        self.ftr_in: self.features[tr_step*self.batch_size:(tr_step+1)*self.batch_size],
                        self.bias_in: self.biases[tr_step*self.batch_size:(tr_step+1)*self.batch_size],
                        self.lbl_in: self.y_train[tr_step*self.batch_size:(tr_step+1)*self.batch_size],
                        self.msk_in: self.train_mask[tr_step*self.batch_size:(tr_step+1)*self.batch_size],
                        self.is_train: True,
                        self.attn_drop: 0.6, self.ffd_drop:0.6})
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            vl_step = 0
            vl_size = self.features.shape[0]

            while vl_step * self.batch_size < vl_size:
                loss_value_vl, acc_vl = self.sess.run([self.loss, self.accuracy],
                    feed_dict={
                        self.ftr_in: self.features[vl_step*self.batch_size:(vl_step+1)*self.batch_size],
                        self.bias_in: self.biases[vl_step*self.batch_size:(vl_step+1)*self.batch_size],
                        self.lbl_in: self.y_val[vl_step*self.batch_size:(vl_step+1)*self.batch_size],
                        self.msk_in: self.val_mask[vl_step*self.batch_size:(vl_step+1)*self.batch_size],
                        self.is_train: False,
                        self.attn_drop: 0.0, self.ffd_drop:0.0})
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1

            print('Training run %d: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                (epoch, train_loss_avg/tr_step, train_acc_avg/tr_step,
                val_loss_avg/vl_step, val_acc_avg/vl_step))
            
            if val_acc_avg/vl_step >= self.vacc_mx or val_loss_avg/vl_step <= self.vlss_mn:
                if val_acc_avg/vl_step >= self.vacc_mx and val_loss_avg/vl_step <= self.vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    self.saver.save(self.sess, self.checkpt_file)
                self.vacc_mx = np.max((val_acc_avg/vl_step, self.vacc_mx))
                self.vlss_mn = np.min((val_loss_avg/vl_step, self.vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == self.patience:
                    print('Early stop! Min loss: ', self.vlss_mn, ', Max accuracy: ', self.vacc_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    return

            train_loss_avg = 0    
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        return "Training complete"

    def get_accuracy(self):
        self.saver.restore(self.sess, self.checkpt_file)
        ts_size = self.features.shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * self.batch_size < ts_size:
            loss_value_ts, acc_ts = self.sess.run([self.loss, self.accuracy],
                feed_dict={
                    self.ftr_in: self.features[ts_step*self.batch_size:(ts_step+1)*self.batch_size],
                    self.bias_in: self.biases[ts_step*self.batch_size:(ts_step+1)*self.batch_size],
                    self.lbl_in: self.y_test[ts_step*self.batch_size:(ts_step+1)*self.batch_size],
                    self.msk_in: self.test_mask[ts_step*self.batch_size:(ts_step+1)*self.batch_size],
                    self.is_train: False,
                    self.attn_drop: 0.0, self.ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        #print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)
        end = time.time()
        print('Total execution time: ', end - self.start)
        return ts_loss/ts_step, ts_acc/ts_step

    def exit(self):
        self.sess.close()    

    def test(self):
        return "testing"

#ray.init()
#gat = GAT_run.remote(dataset)
#a = gat.test.remote()
#print(ray.get(a))
