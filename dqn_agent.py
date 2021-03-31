
import numpy as np
import Environment
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import random
import math
import os
import cv2
import copy
from collections import Counter
import pandas as pd
import csv
from collections import deque

from scipy.stats import norm
import matplotlib.pyplot as plt
from math import exp







TRAIN_FREQUENCY = 4

MAX_MEMORY_SIZE = 10000
# hyperparameters
height = 50
# width = 100
width = 200
# n_obs = 100 * 50  # dimensionality of observations
n_obs = 200 * 50
h = 200  # number of hidden layer neurons
n_actions = 4  # number of available actions
learning_rate = 0.00025
gamma = .99  # discount factor for reward

batch_size = 32

learn_start = 1000
learning_rate_minimum = 0.00025,
learning_rate_decay_step = 5 * 10000,
learning_rate_decay = 0.96,
target_q_update_step = 1000
decay = 0.99  # decay rate for RMSProp gradients
save_path = 'models_Attemp800\Attempt800'
# save_path = 
INITIAL_EPSILON = 1

# gamespace
display = False
training = True

game=Environment.GameV1()
game.populateGameArray()



def clipped_error(x):
  # Huber loss
  try:
    return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
  except:
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)





def conv2d(x,
           output_dim,
           kernel_size,
           stride,
           initializer=tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu,
           data_format='NHWC',
           padding='VALID',
           name='conv2d'):

    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]
        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)

        #input = [batch, in_height, in_width, in_channels]
        # filters = [filter_height, filter_width, in_channels, out_channels]
        # tf.nn.conv2d(input, filters, strides, padding, data_format='NHWC', dilations=None, name=None)
        conv = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)

        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, b, data_format)

        if activation_fn != None:
            out = activation_fn(out)
        return out, w, b











def linear(input_, output_size, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear'):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(name):
    w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
        tf.random_normal_initializer(stddev=stddev))
    b = tf.get_variable('bias', [output_size],
        initializer=tf.constant_initializer(bias_start))

    out = tf.nn.bias_add(tf.matmul(input_, w), b)

    if activation_fn != None:
        return activation_fn(out), w, b
    else:
        return out, w, b






def setup_training_step_count():
    with tf.variable_scope('step'):
        ## tf Variable  for calculating number of steps
        step_op = tf.Variable(0, trainable=False, name='step')

        ## tf .place holder shape  = None, make sure to run it with feed dict othervise gives error
        step_input = tf.placeholder('int32', None, name='step_input')

        ## step_assign_op 
        step_assign_op = step_op.assign(step_input)

    return step_op, step_input, step_assign_op





class dqn_Model():
    def __init__(self, network_scope_name, sess):
        self.network_scope_name = network_scope_name
        self.sess = sess
        self.importance_in = tf.placeholder(tf.float32, shape=[None])
        # self.importance_in = tf.placeholder(tf.float32, shape=[None])
        pass

    def forward_graph(self):
        self.w = {} ## dictionary 

        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        with tf.variable_scope(self.network_scope_name):
            self.s_t = tf.placeholder(dtype=tf.float32,
                                      shape=[None, height, width, 1], name='s_t')
            self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
                                                             32, [8, 8], [4, 4], initializer, activation_fn,
                                                             "NHWC", name='l1')
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
                                                             64, [4, 4], [2, 2], initializer, activation_fn,
                                                             "NHWC", name='l2')
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
                                                             64, [3, 3], [1, 1], initializer, activation_fn,
                                                             "NHWC", name='l3')
            ## for fully connected layer we flatten the last one and send it as an ANN

            self.l3_flat = flatten(self.l3)
            self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = \
                linear(self.l3_flat, 512, activation_fn=activation_fn, name='value_hid')

            self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = \
                linear(self.l3_flat, 512, activation_fn=activation_fn, name='adv_hid')

            self.value, self.w['val_w_out'], self.w['val_w_b'] = \
                linear(self.value_hid, 1, name='value_out')

            self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
                linear(self.adv_hid, n_actions, name='adv_out')

            # Average Dueling
            self.q = self.value + (self.advantage -
                                   tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))

            self.a_action = tf.argmax(self.q, axis=1)



    def select_q_graph(self):
        self.q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
        ## tf.gather_nd  will take the list numbers and put them as array indices for slicing
        self.q_with_idx = tf.gather_nd(self.q, self.q_idx)





    def train_graph(self):
        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.target_q_t = tf.placeholder(shape=[None], dtype=tf.float32, name='target_q_t')
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='action')

        ## categorical values actions_onehot
        self.actions_onehot = tf.one_hot(self.actions, n_actions, dtype=tf.float32, name='action_one_hot')

        ## q_acted is the current prediction

        self.q_acted = tf.reduce_sum(tf.multiply(self.q, self.actions_onehot), axis=1, name='q_acted')
        self.error = self.target_q_t - self.q_acted
        self.delta = tf.square(self.target_q_t - self.q_acted)
        #self.global_step = tf.Variable(0, trainable=False)
        # apply huber loss to clipping the error, and derivative

        # self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
        # prioritized_experience_replay
        self.loss = tf.reduce_mean(tf.multiply(tf.square(self.error), self.importance_in))


        self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')

        self.learning_rate_op = tf.maximum(learning_rate_minimum,
                                           tf.train.exponential_decay(
                                               learning_rate,
                                               self.learning_rate_step,
                                               learning_rate_decay_step,
                                               learning_rate_decay,
                                               staircase=True))

        #Optimizer
        self.optim = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate, momentum=0.95, epsilon=0.01)

        tf_grads = self.optim.compute_gradients(self.loss,  var_list=tf.trainable_variables())
        self.train_op = self.optim.apply_gradients(tf_grads)

        grad_summaries = []
        for g, v in tf_grads:
            if g is not None:
                #grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                #grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        loss_summary = tf.summary.scalar("hubber_loss ", self.loss)
        #learning_rate_ = tf.summary.scalar("learning_rate", self.learning_rate_op)

        self.train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])



    def summary_graph(self):
        q_summary = []
        avg_q = tf.reduce_mean(self.q, 0)
        for idx in range(n_actions):
            q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))

        self.q_summary = tf.summary.merge(q_summary, 'q_summary')

        self.win_rate_holder = tf.placeholder("float32", None, name="running_win_rate")
        self.win_rate_op = tf.summary.scalar("win_rate", self.win_rate_holder)

        scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
                               'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game',
                               'training.learning_rate']

        self.summary_placeholders = {}
        self.summary_ops = {}

        for tag in scalar_summary_tags:
            self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
            self.summary_ops[tag] = tf.summary.scalar("%s" % (tag),
                                                      self.summary_placeholders[tag])

        histogram_summary_tags = ['episode.rewards', 'episode.actions']

        for tag in histogram_summary_tags:
            self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
            self.summary_ops[tag] = tf.summary.histogram(tag, self.summary_placeholders[tag])

        train_summary_dir = os.path.join(save_path, "summaries", "train")
        self.writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)


    def update_network(self):
        if self.network_scope_name == "target":
            self.w_input = {}
            self.w_assign_op = {}
            for name in self.w.keys():
                self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
                self.w_assign_op[name] = self.w[name].assign(self.w_input[name])




## there are two networks main and target and we copy all parameters from main to target
def update_target_network(target_network, main_network):
    for name in target_network.w.keys():
        target_network.w_assign_op[name].eval({target_network.w_input[name] : main_network.w[name].eval()})







# class experience_buffer():
#     def __init__(self, buffer_size=50_000):
#         self.buffer = []
#         self.buffer_size = buffer_size
#         self.priorities = deque(maxlen=buffer_size)

#     def add(self, experience):
#         if len(self.buffer) + len(experience) >= self.buffer_size:
#             self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
#         self.buffer.extend(experience)
#         self.priorities.append(max(self.priorities,default=1)) ## initial priority value

#     def get_probabilities(self, priority_scale):
#         scaled_priorities = np.array(self.priorities) ** priority_scale
#         sample_probablities  = scaled_priorities / sum(scaled_priorities)
#         return sample_probablities
        
#     def get_importance(self, probabilities):
#         importance = 1/len(self.buffer) * 1/probabilities
#         importance_normalized = importance / max(importance)
#         return importance_normalized  

#     def sample(self, size, priority_scale = 1.0):
#         sample_size = min(len(self.buffer), size)
#         sample_probs = self.get_probabilities(priority_scale)
#         sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
#         samples = np.array(self.buffer)[sample_indices]
#         importance = self.get_importance(sample_probs[sample_indices])
#         return map(list, zip(*samples)), importance, sample_indices
#         # return np.reshape(np.array(random.sample(self.buffer, size)), [size, 7]), importance, sample_indices
         
#     def set_priorities(self, indices, errors, offset=0.1):
#         for i,e in zip(indices, errors):
#             self.priorities[i] = abs(e) + offset

class experience_buffer():
    def __init__(self,maxlen =50_000):
        # self.buffer = []
        self.buffer = deque(maxlen=maxlen)
        # self.buffer_size = buffer_size
        self.priorities = deque(maxlen=maxlen)

    def add(self, experience):
        # if len(self.buffer) + len(experience) >= self.buffer_size:
        #     self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        # self.buffer.extend(experience)
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities,default=1)) ## initial priority value

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probablities  = scaled_priorities / sum(scaled_priorities)
        return sample_probablities
        
    def get_importance(self, probabilities):
        importance = 1/len(self.buffer) * 1/probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized  

    def sample(self, size, priority_scale = 1.0):
        sample_size = min(len(self.buffer), size)
        sample_probs = self.get_probabilities(priority_scale)

        # print("length_self.buffer","sample_size","sample_probs",\
        # len(self.buffer),sample_size,sample_probs)

        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        samples = np.array(self.buffer)[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        return map(list, zip(*samples)), importance, sample_indices
        # return np.reshape(np.array(random.sample(self.buffer, size)), [size, 7]), importance, sample_indices
         
    def set_priorities(self, indices, errors, offset=0.1):
        for i,e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset


# def displayImage(image):
#     __debug_diff__ = False
#     if __debug_diff__:
#         cv2.namedWindow("debug image")
#         cv2.imshow('debug image', image)
#         cv2.waitKey(2000)
#         cv2.destroyWindow("debug image")






def prepro(I):
    """ prepro 210x160x3 uint8 frame into 5000 (50x100) 1D float vector """
    I = I[::4, ::2]  # downsample by factor of 2
    return I.astype('float32')






def count_win_percentage(win_loss):
    count = 0
    for win in win_loss:
        if win == 1:
            count +=1
    return 100 * count / len(win_loss)


def restore_model(sess):
    episode_number = 0

    # try load saved model
    saver = tf.train.Saver(tf.global_variables())
    load_was_success = True  # yes, I'm being optimistic
    try:
        save_dir = '/'.join(save_path.split('/')[:-1])
        ckpt = tf.train.get_checkpoint_state(save_dir)
        load_path = ckpt.model_checkpoint_path
        saver.restore(sess, load_path)

    except:
        print(
            "no saved model to load. starting new session")
        tf.global_variables_initializer().run()
        load_was_success = False
    else:
        print(
            "loaded model: {}".format(load_path))
        #saver = tf.train.Saver(tf.global_variables())
        episode_number = int(load_path.split('-')[-1])

    return saver, episode_number, save_dir




def func(x):
  n = norm(6, 2)
  if (x>=0):
    return n.pdf(x) * (1/n.pdf(6))
  else:
    return -1


def func2(x):
    if (0<=x<=6):
      return (exp(abs(x)-6))
    elif(x>6):
      return (exp(-abs(x)+6))  
    else :
      return -1


def func_1_batch_vel(batch_vel):
    func_1_vel = copy.deepcopy(batch_vel)
    func_1_vel = [func(x) for x in func_1_vel]
    return func_1_vel

def func_2_batch_vel(batch_vel):
    func_2_vel = copy.deepcopy(batch_vel)
    func_2_vel = [func2(x) for x in func_2_vel]
    return func_2_vel


def train(sess):

    # create two graphs
    main_q_net = dqn_Model("main", sess)
    target_q_net = dqn_Model("target", sess)

    # create graphs, although there are two graphs for main and target networks, but they all belong to the same session
    # hence, the model save and restore will save them all and restore them all
    # however, the scope name will differentiate the main network from the target network
    main_q_net.forward_graph() ## creating all the layers
    main_q_net.train_graph()  ## training set up

    target_q_net.forward_graph() #type of network
    target_q_net.select_q_graph() ## select q with idx
    target_q_net.update_network() 

    ## variable, tensor, variable = tensor 0 is assigned
    step_op, step_input, step_assign_op = setup_training_step_count()


    main_q_net.summary_graph() # create the summary graph last, so ops will be in tensorboard

    # restore the model with previously saved weights

    saver, start_episode_number, save_dir = restore_model(sess)

    step = step_op.eval(session=sess)

    # train_summary_dir = os.path.join(save_dir, "summaries", "train")
    # train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    #epsilon = 1
    epsilon = math.pow(0.5, start_episode_number/3000)


    wait_time = 1
    waited_time = 0

    running_reward = None
    reward_sum = 0
    observation = np.zeros(shape=(200, 300))

    win_loss = []
    # training loop
    replay_buffer = experience_buffer()

    # preprocess the observation, set input to network to be difference image
    s_t = prepro(observation)



    episode_number = start_episode_number

    ## store the episods
    episodeBuffer = experience_buffer()

    ## continuous save the win rate
    last_saved_win_rate = 0

    sess.graph.finalize() ## freeze the graph read only

    ## time_stamps_survived = 0
    time_stamps_survived = 0
    ## velocity_average
    vel_sum = 0
    ## average reward per episode
    episode_reward_sum = 0
    ## maximum_velocity_in_episode
    max_vel = 0
    ##action frequency counter
    action_freq = Counter()

    while True: # looping over every step. episode consists of many steps till end of a game?
        ## first iteration

        if waited_time < wait_time:
            action = 0
            waited_time += 1
            ## the velocity is velocity * probability_status
            observation, reward, smallreward, done, velocity = game.runGame(action, False)
            ## first action is 0;
        else:
            # perform epsilon greedy for explorationa nd exploitation
            if random.random() < epsilon:
                action = random.randint(0, n_actions-1)
            else:
                feed = {main_q_net.s_t:np.reshape(s_t, [-1, height, width, 1]) }
                action = sess.run(main_q_net.a_action, feed)[0] ## we get the action here

            # roll out a step
            observation, reward, smallreward, done, velocity = game.runGame(action, False)
            action_freq[action]+=1
            ## VELOCITIES maximum and vel sum
            # print("velocity",velocity)

            absolute_velocity = abs(velocity)

            max_vel = max(absolute_velocity,max_vel)
            vel_sum += absolute_velocity
            episode_reward_sum += func2(velocity) ## can be negative confirm is function is required here because experience replay
            ## has the func

            s_t_plus_1 = prepro(observation)

            # save everything about this step into the episode buffer
            # episodeBuffer.add(np.reshape(np.array([s_t, action, reward, smallreward, s_t_plus_1, done, velocity]), [1, 7]))
            episodeBuffer.add((s_t, action, reward, smallreward, s_t_plus_1, done, velocity))
            #reset state variable for next step
            s_t = s_t_plus_1


            # parameter update
            ##learn_start  =1000
            ## start_episode_number = 0 if or start from saved number 
            if episode_number-start_episode_number > learn_start: # wait till enough in the play buffer
                # sample a batch from the replay buffer
                # ready to train the networks
                if step % TRAIN_FREQUENCY == 0:
                    ##uniform sampling


                    # train_batch = replay_buffer.sample(batch_size)
                    # train_batch , importance, sample_indices = replay_buffer.sample(batch_size,priority_scale=1.0)
                    (batch_s_t,batch_action,batch_reward,batch_smallreward,batch_s_t_plus_1,batch_done,batch_velocity), importance, indices = replay_buffer.sample(batch_size)


                    # unpack the samples
                    # batch_s_t = np.expand_dims(np.stack(train_batch[:, 0]), -1)
                    # batch_action = train_batch[:, 1]
                    # batch_reward = np.expand_dims(np.stack(train_batch[:, 2]), -1)
                    # batch_smallreward = train_batch[:, 3]
                    # batch_s_t_plus_1 = np.expand_dims(np.stack(train_batch[:, 4]), -1)
                    # batch_done = train_batch[:, 5]
                    # batch_velocity = train_batch[:,6]
                    batch_s_t = np.expand_dims(np.stack(batch_s_t), -1)
                    batch_reward = np.expand_dims(np.stack(batch_reward), -1)
                    batch_s_t_plus_1 = np.expand_dims(np.stack(batch_s_t_plus_1), -1)

                    ## importance we will get it and then back propogation

                    # get a_t_plus_1 from the main net
                    pred_action = main_q_net.a_action.eval({main_q_net.s_t: batch_s_t_plus_1})
                    # get q_t_plus_1 from the target net (not max, a main diff from the vallina dqn)
                    q_t_plus_1_with_pred_action = target_q_net.q_with_idx.eval({target_q_net.s_t: batch_s_t_plus_1,
                                                                                target_q_net.q_idx: \
                                                                                    [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]})
                    # compute q estimates
                    
                    #target_q_t = (1.0 - batch_done)*gamma * q_t_plus_1_with_pred_action + batch_smallreward
                    ## batch_velocity has negative and positive  for done win/loss
                    # and positive scores or velocities for not done, middle time steps
                    # target_q_t = batch_velocity + batch_smallreward
                    # target_q_t = batch_velocity +  gamma * q_t_plus_1_with_pred_action ## DOUBLE DQN
                    # target_q_t = rewards + gamma * q_t_plus_1_with_pred_action DOUBLE DQN



                    new_reward_1 = func_2_batch_vel(batch_velocity) ## reward for batches as a function of velocity

                    target_q_t = new_reward_1


                    # prepare input for backprop on main network
                    feed = {main_q_net.s_t: batch_s_t,
                            main_q_net.actions: batch_action,
                            main_q_net.target_q_t: target_q_t,
                            main_q_net.learning_rate_step: step,
                            main_q_net.importance_in: importance**(1-epsilon)
                            }

                    _, q_t,errors, loss, train_summaries, q_summaries = sess.run([main_q_net.train_op, main_q_net.q, main_q_net.error,\
                     main_q_net.loss, main_q_net.train_summary_op, main_q_net.q_summary],\
                                                             feed_dict=feed)

                    replay_buffer.set_priorities(indices, errors)

                    main_q_net.writer.add_summary(train_summaries, step)
                    main_q_net.writer.add_summary(q_summaries, step)

                # update target network, and save the model to hd
                if step % target_q_update_step == target_q_update_step -1:
                    # update the target network
                    update_target_network(target_q_net,main_q_net)

                    # persist models only when win_rate is better
                    if win_rate > last_saved_win_rate:
                        step_assign_op.eval({step_input: step})
                        saver.save(sess, save_path, global_step=episode_number)

                        print("SAVED MODEL #step {}, episode{}".format(step, episode_number))
                        last_saved_win_rate = win_rate
            time_stamps_survived+=1
            step += 1
            if done: #end of episode
                #reset
                waited_time = 0



                win_loss.append(reward)
                while len(win_loss)>100:
                    win_loss.pop(0)
                # add the samples for the episode to the replay buffer
                # TODO: should we adjust sample rewards based on win or loss of the episode?
                sample_count = len(episodeBuffer.buffer)
                # keep only 2nd half of the samples, so that the replay buffer will have more negative samples
                # replay_buffer.add(episodeBuffer.buffer[1::4]
                #                   + [copy.deepcopy(episodeBuffer.buffer[-1]) for _ in range(20)])
                for i in range(1,len(episodeBuffer.buffer),4):
                    replay_buffer.add(episodeBuffer.buffer[i])
                for _ in range(20):
                    replay_buffer.add(episodeBuffer.buffer[-1])
                # update running reward
                epsilon = math.pow(0.5, episode_number / 3000)
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01





                # print progress console
                win_rate = count_win_percentage(win_loss) ## an float value
                # print('\tep {}: running_reward: {}, won {:.2f} %'.format(episode_number, running_reward, win_rate))
                
                
                
                ## calculate_average_vel
                avg_vel = float(vel_sum/time_stamps_survived) ## always positive
                avg_episode_reward = float(episode_reward_sum/time_stamps_survived)




                f = open( 'fuel_eff_2_rel.txt', 'a+')
                f.write(str(episode_number) + " " \
                + str(time_stamps_survived) + " " + str(reward) \
                + " " + str(avg_episode_reward)\
                + " " + str(action_freq)\
                + " " + str(max_vel) +" " + str(avg_vel) +" "+ str(epsilon) + " "+ str(win_rate)+ '\n')  
                f.close()




                ## clear all variables for new episode
                time_stamps_survived = 0
                max_vel = 0
                avg_vel = 0
                vel_sum = 0
                episode_reward_sum = 0
                action_freq.clear()





                # write the win_rate for tensorboard
                step_assign_op.eval({step_input:step})
                win_rate_summary = sess.run(main_q_net.win_rate_op, feed_dict={main_q_net.win_rate_holder:win_rate})
                main_q_net.writer.add_summary(win_rate_summary, step)

                episode_number += 1  # the Next Episode

                reward_sum = 0
                # reset the the episode buffer for next episode
                episodeBuffer = experience_buffer()

def inference(sess):
    observation = np.zeros(shape=(200, 300))
    prev_x = None
    #create forward graph
    main_q_net = dqn_Model("main", sess)
    main_q_net.forward_graph()

    # restore the model with previously saved weights
    #step_op, step_input, step_assign_op = setup_training_step_count()
    saver, episode_number, save_dir = restore_model(sess)
    #step = step_op.eval(session=sess)
    wait_time = 1
    waited_time = 0


    win_loss = []
    reward_sum = 0
    running_reward = None
    while True:
        # preprocess the observation, set input to network to be difference image
        s_t = prepro(observation)

        if waited_time < wait_time:
            action = 0
            waited_time += 1
        else:
            feed = {main_q_net.s_t: np.reshape(s_t, [-1, height, width, 1])}
            action = sess.run(main_q_net.a_action, feed)[0];

        observation, reward, smallreward, done, velocity = game.runGame(action, False)

        reward_sum += reward

        if done:
            # reset
            waited_time = 0
            episode_number += 1

            win_loss.append(reward)
            while len(win_loss) > 100:
                win_loss.pop(0)

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            # print(
            #     '\tep {}: reward: {}, won {:.2f} %'.format(episode_number, reward_sum, count_win_percentage(win_loss)))

            reward_sum = 0

def main():

    # tf graph initialization
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()



    if training:
        train(sess)
    else:
        inference(sess)


if __name__=="__main__":
    main()



