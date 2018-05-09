import time
import numpy as np
import tensorflow as tf
import ray

from models import GAT
from utils import process

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--redis_address', '-r', default=None, type=str, help='pass in the redis_address if running on a cluster, otherwise omit')
args = parser.parse_args()

checkpt_file = 'pre_trained/cora/mod_cora.ckpt'

dataset = 'cora'

# If redis_address provided, initialize ray with that address
if args.redis_address:
    ray.init(redis_address=args.redis_address)
else:
    ray.init()

# training params
batch_size = 1
nb_epochs = 100000
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = GAT

print('Starting timer')
start = time.time()
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

class CalculateAccuracy(object):
    def __init__(self):
        pass
        # self.info_finished = info_finished
        # self.adj_PLACEHOLDER, self.features_PLACEHOLDER, self.y_train_PLACEHOLDER, self.y_val_PLACEHOLDER, self.y_test, self.train_mask_PLACEHOLDER, self.val_mask_PLACEHOLDER, self.test_mask = process.load_data(dataset)
        # self.nb_nodes = self.features_PLACEHOLDER.shape[0]
        # self.ft_size = self.features_PLACEHOLDER.shape[1]
        # self.nb_classes = self.y_train_PLACEHOLDER.shape[1]
    
    def test_accuracy(self, info_finished):
        self.info_finished = info_finished
        self.adj_PLACEHOLDER, self.features_PLACEHOLDER, self.y_train_PLACEHOLDER, self.y_val_PLACEHOLDER, self.y_test, self.train_mask_PLACEHOLDER, self.val_mask_PLACEHOLDER, self.test_mask = process.load_data(dataset)
        self.nb_nodes = self.features_PLACEHOLDER.shape[0]
        self.ft_size = self.features_PLACEHOLDER.shape[1]
        self.nb_classes = self.y_train_PLACEHOLDER.shape[1]
        with tf.Graph().as_default():
            with tf.name_scope('input'):
                # ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.nb_nodes, self.ft_size))
                # bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.nb_nodes, self.nb_nodes))
                #TODO: Find logits size
                logits_size = (len(self.info_finished), self.nb_nodes, self.nb_classes)
                logits_list = tf.placeholder(dtype=tf.float32, shape=logits_size)
                lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, self.nb_nodes, self.nb_classes))
                msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, self.nb_nodes))
                # attn_drop = tf.placeholder(dtype=tf.float32, shape=())
                # ffd_drop = tf.placeholder(dtype=tf.float32, shape=()))
                # is_train = tf.placeholder(dtype=tf.bool, shape=())

            # logits = model.inference(ftr_in, self.nb_classes, self.nb_nodes, is_train,
            #                             attn_drop, ffd_drop,
            #                             bias_mat=bias_in,
            #                             hid_units=hid_units, n_heads=n_heads,
            #                             residual=residual, activation=nonlinearity)
            
            # logits should already be shapped properly
            # log_resh = tf.reshape(logits, [-1, self.nb_classes])
            lab_resh = tf.reshape(lbl_in, [-1, self.nb_classes])
            msk_resh = tf.reshape(msk_in, [-1])
            accuracy = model.masked_accuracy_multiple(logits_list, lab_resh, msk_resh)
            
            # TODO set the logits here
            print("self.info_finshed, ", self.info_finished)
            self.logits = [info[1] for info in self.info_finished]
            self.logits_list = np.stack(self.logits)
            
            with tf.Session() as sess:
                ts_size = self.features_PLACEHOLDER.shape[0]
                ts_step = 0
                ts_loss = 0.0
                ts_acc = 0.0
                acc_ts = sess.run([accuracy],
                    feed_dict={
                        logits_list: self.logits_list,
                        lbl_in: self.y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                        msk_in: self.test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                ts_acc += acc_ts
                ts_step += 1
                
                print('Test accuracy:', ts_acc/ts_step)
                end = time.time()
                print('Total execution time: ', end - start)
                sess.close()

class doGAT(object):
    def __init__(self, index, dataset):
        pass
        # self.adj, self.features, self.y_train, self.y_val, self.y_test, self.train_mask, self.val_mask, self.test_mask = process.load_data(dataset)

        # self.adj = self.adj[index*1354:(index+1)*1354, index*1354:(index+1)*1354]
        # self.features = self.features[index*1354:(index+1)*1354]
        # self.y_train = self.y_train[index*1354:(index+1)*1354]
        # self.y_val = self.y_val[index*1354:(index+1)*1354]
        # self.y_test = self.y_test[index*1354:(index+1)*1354]
        # self.train_mask = self.train_mask[index*1354:(index+1)*1354]
        # self.val_mask = self.val_mask[index*1354:(index+1)*1354]
        # self.test_mask = self.test_mask[index*1354:(index+1)*1354]

        # for i in range(100, 2708):
        #     self.features[i] = np.zeros(1433);
        #     self.y_train[i] = 0;
        #     self.y_val[i] = 0;
        #     self.y_test[i] = 0;
        #     self.train_mask[i] = 0;
        #     self.val_mask[i] = 0;
        #     self.test_mask[i] = 0;
        #     # for j in range(100, 2708):
        #     #     self.adj[i,j] = 0;

        # self.nb_nodes = self.features.shape[0]
        # self.ft_size = self.features.shape[1]
        # self.nb_classes = self.y_train.shape[1]
        #
        # self.features, self.spars = process.preprocess_features(self.features)

    def magic(self, index):
        self.adj, self.features, self.y_train, self.y_val, self.y_test, self.train_mask, self.val_mask, self.test_mask = process.load_data_new(dataset, index)
        self.nb_nodes = self.features.shape[0]
        self.ft_size = self.features.shape[1]
        self.nb_classes = self.y_train.shape[1]

        self.features, self.spars = process.preprocess_features(self.features)

        self.adj = self.adj.todense()

        if(index == 0):
            print("&&&&&")
        else:
            print("*****")
        print("adj")
        print(self.adj.shape)
        print("features")
        print(self.features.shape)
        print("ytrain")
        print(self.y_train.shape)
        print("yval")
        print(self.y_val.shape)
        print("ytest")
        print(self.y_test.shape)
        print("trainmask")
        print(self.train_mask.shape)
        print("valmask")
        print(self.val_mask.shape)
        print("testmask")
        print(self.test_mask.shape)
        if(index == 0):
            print("&&&&&")
        else:
            print("*****")

        self.features = self.features[np.newaxis]
        self.adj = self.adj[np.newaxis]
        self.y_train = self.y_train[np.newaxis]
        self.y_val = self.y_val[np.newaxis]
        self.y_test = self.y_test[np.newaxis]
        self.train_mask = self.train_mask[np.newaxis]
        self.val_mask = self.val_mask[np.newaxis]
        self.test_mask = self.test_mask[np.newaxis]

        self.biases = process.adj_to_bias(self.adj, [self.nb_nodes], nhood=1)

        with tf.Graph().as_default():
            with tf.name_scope('input'):
                ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.nb_nodes, self.ft_size))
                bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.nb_nodes, self.nb_nodes))
                lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, self.nb_nodes, self.nb_classes))
                msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, self.nb_nodes))
                attn_drop = tf.placeholder(dtype=tf.float32, shape=())
                ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
                is_train = tf.placeholder(dtype=tf.bool, shape=())

            logits = model.inference(ftr_in, self.nb_classes, self.nb_nodes, is_train,
                                        attn_drop, ffd_drop,
                                        bias_mat=bias_in,
                                        hid_units=hid_units, n_heads=n_heads,
                                        residual=residual, activation=nonlinearity)
            log_resh = tf.reshape(logits, [-1, self.nb_classes])
            lab_resh = tf.reshape(lbl_in, [-1, self.nb_classes])
            msk_resh = tf.reshape(msk_in, [-1])
            loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
            accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

            train_op = model.training(loss, lr, l2_coef)

            saver = tf.train.Saver()

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            vlss_mn = np.inf
            vacc_mx = 0.0
            curr_step = 0

            with tf.Session() as sess:
                sess.run(init_op)

                train_loss_avg = 0
                train_acc_avg = 0
                val_loss_avg = 0
                val_acc_avg = 0

                for epoch in range(nb_epochs):
                    tr_step = 0
                    tr_size = self.features.shape[0]

                    while tr_step * batch_size < tr_size:
                        _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                            feed_dict={
                                ftr_in: self.features[tr_step*batch_size:(tr_step+1)*batch_size],
                                bias_in: self.biases[tr_step*batch_size:(tr_step+1)*batch_size],
                                lbl_in: self.y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                                msk_in: self.train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                                is_train: True,
                                attn_drop: 0.6, ffd_drop: 0.6})
                        train_loss_avg += loss_value_tr

                        train_acc_avg += acc_tr
                        tr_step += 1

                    vl_step = 0
                    vl_size = self.features.shape[0]

                    while vl_step * batch_size < vl_size:
                        loss_value_vl, acc_vl = sess.run([loss, accuracy],
                            feed_dict={
                                ftr_in: self.features[vl_step*batch_size:(vl_step+1)*batch_size],
                                bias_in: self.biases[vl_step*batch_size:(vl_step+1)*batch_size],
                                lbl_in: self.y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                                msk_in: self.val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                                is_train: False,
                                attn_drop: 0.0, ffd_drop: 0.0})
                        val_loss_avg += loss_value_vl
                        val_acc_avg += acc_vl
                        vl_step += 1

                    print('%d: Training run %d: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                            (index, epoch, train_loss_avg/tr_step, train_acc_avg/tr_step,
                            val_loss_avg/vl_step, val_acc_avg/vl_step), flush=True)

                    if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                        if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                            vacc_early_model = val_acc_avg/vl_step
                            vlss_early_model = val_loss_avg/vl_step
                            saver.save(sess, checkpt_file)
                        vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                        vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                        curr_step = 0
                    else:
                        curr_step += 1
                        if curr_step == patience:
                            print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx, flush=True)
                            print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model, flush=True)
                            break

                    train_loss_avg = 0
                    train_acc_avg = 0
                    val_loss_avg = 0
                    val_acc_avg = 0

                saver.restore(sess, checkpt_file)
                ts_step = 0
                logits = sess.run([log_resh],
                    feed_dict={
                            ftr_in: self.features[ts_step*batch_size:(ts_step+1)*batch_size],
                            bias_in: self.biases[ts_step*batch_size:(ts_step+1)*batch_size],
                            lbl_in: self.y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                            msk_in: self.test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                            is_train: False,
                            attn_drop: 0.0, ffd_drop: 0.0})
                            
                sess.close()
                end = time.time()
                print('Total execution time: ', end - start)
                
                return (index, logits)

                # ts_size = self.features.shape[0]
                # ts_step = 0
                # ts_loss = 0.0
                # ts_acc = 0.0
                # print("ts_size: ", ts_size)
                # while ts_step * batch_size < ts_size:
                #     loss_value_ts, acc_ts = sess.run([loss, accuracy],
                #         feed_dict={
                #             ftr_in: self.features[ts_step*batch_size:(ts_step+1)*batch_size],
                #             bias_in: self.biases[ts_step*batch_size:(ts_step+1)*batch_size],
                #             lbl_in: self.y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                #             msk_in: self.test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                #             is_train: False,
                #             attn_drop: 0.0, ffd_drop: 0.0})
                #     ts_loss += loss_value_ts
                #     ts_acc += acc_ts
                #     ts_step += 1
                #     print("iteration", ts_loss, " ", ts_acc, " ", ts_step)
                # 
                # print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)
                # end = time.time()
                # print('Total execution time: ', end - start)
                # sess.close()
                # return (ts_acc/ts_step, logits)

remote_network = ray.remote(doGAT)
actor_list = [remote_network.remote(i, dataset) for i in range(2)]
things = [actor_list[i].magic.remote(i) for i in range(2)]
info_finished_ids, _ = ray.wait(things)
print("CROSSED WAIT WTF")
accuracy_remote = ray.remote(CalculateAccuracy)
accuracy_actor = accuracy_remote.remote()
accuracy_id = accuracy_actor.test_accuracy.remote(ray.get(info_finished_ids))

accuracy = ray.get(accuracy_id)
print("The Accuracy after Ensembling is: ", accuracy)
# gradients_list = ray.get(things)
# print(gradients_list)
