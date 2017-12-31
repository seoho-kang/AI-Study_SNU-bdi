import sys
import numpy as np
import random
import env_parking as env
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class RL_agent(object):
    
    def __init__(self):
        
        # defining the environment and the figure to draw
        self.fig = plt.figure()
        if sys.argv[-1] == '--big': # bigger environment
            self.env = env.Parking(self.fig, size_board=(16,16), rewards={(0, 12): -1, (1, 12): -1, (2, 12): -1, (7, 12): -1, (7, 13): -1, (7, 14): -1, (7, 15): -1, (2, 15): 1},
                base_reward=0, obstacles=[], pCorrect = 0.99)
        else:
            self.env = env.Parking(self.fig)

        # defining values(state-value, action-value) and policy
        self.num_actions = self.env.num_actions
        self.actions = range(self.num_actions)
        self.val_Q = np.clip(np.random.normal(size = (self.env.num_states, self.num_actions), scale = 0.01), -0.1, 0.1)
        self.val_V = np.clip(np.random.normal(size = (self.env.num_states), scale = 0.01), -0.1, 0.1)
        self.mat_policy = np.zeros([self.env.num_states, self.num_actions])
        self.visit_actions = np.zeros([self.env.num_states, self.num_actions])
        self.vec_policy = np.argmax(self.val_Q, axis = 1)
        
        # parameters used for learning
        self.acc_reward = 0.
        self.epsilon = 0.5
        self.gammalist = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999]
        self.idx_gamma = len(self.gammalist) - 4
        self.gamma = self.gammalist[self.idx_gamma]
        self.idx_it = 0.
        #self.size_step = 0.1
        
        ## drawing the environment on the figure ##
        plt.xticks([],[])
        plt.yticks([],[])
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show(block=False)
        ####

    # initialize values
    def init_vals(self):
        self.val_Q = np.clip(np.random.normal(size = (self.env.num_states, self.num_actions), scale = 0.01), -0.1, 0.1)
        self.val_V = np.clip(np.random.normal(size = (self.env.num_states), scale = 0.01), -0.1, 0.1)

    # initialize policy
    def init_policy(self):
        self.mat_policy = np.zeros([self.env.num_states, self.num_actions])
        self.vec_policy = np.argmax(self.val_Q, axis = 1)
    
    # reset everything except the agent's current state
    def init_everything(self):
        self.init_vals()
        self.init_policy()
        self.visit_actions = np.zeros([self.env.num_states, self.num_actions])
        self.idx_it = 0.
        self.acc_reward = 0.
        
    # update the agent's policy according to its action-value table
    def update_policy(self, prt = False):
        self.vec_policy = np.argmax(self.val_Q, axis = 1)
        self.mat_policy = np.zeros([self.env.num_states, self.num_actions])
        
        for i in range(len(self.vec_policy)):
            self.mat_policy[i, self.vec_policy[i]] = 1
                    
        if prt: 
            print self.mat_policy
            print self.vec_policy
        
    # learn values using dynamic programming    
    def learn_DP(self, num_epoch = 100, num_iteration = 100, thr_converge = 1e-2, prt = False):
        time_prev = time.time() # record the time when Dynamic Programming started
        print "Dynamic Programming started"

        # update policy, iterating policy evaluation
        old_valQ_it = np.copy(self.val_Q)
        for it in range(num_iteration):            
            old_valQ_ep = np.copy(self.val_Q)

            # learn state values of the current policy
            for ep in range(num_epoch):
                self.val_V = np.sum(np.multiply(self.mat_policy, self.val_Q), axis = 1) # updated state values
                exp_valQ = np.zeros([self.env.num_states, self.num_actions]) 
                
                for i in range(self.num_actions):
                    exp_valQ[:, i] = self.env.mat_transition[i].dot(self.val_V) # expected action values
                self.val_Q = self.env.mat_rewards + self.gamma * exp_valQ # update action values

                # check convergence based on difference sum of state values
                if np.sum(np.abs(self.val_Q - old_valQ_ep)) < thr_converge: 
                    print "converged, iteration stopped"
                    break
                old_valQ_ep = np.copy(self.val_Q)

            if np.sum(np.abs(self.val_Q - old_valQ_it)) < thr_converge: 
                print "converged, policy update stopped"
                break

            old_valQ_it = np.copy(self.val_Q)
            self.update_policy(prt) # update policy according to the learnt action values
            if it == (num_iteration - 1): print "ended without convergence"
        time_next = time.time() # record the time when Dynamic Programming ended
        print "Learning by dynamic programming done; {} seconds passed".format(time_next - time_prev)
        
    # learn state values using TD(0) learning
    def learn_TD0_step(self, iteration):
        curr_state = self.env.state
        curr_action = self.vec_policy[curr_state] # choose an action according to the agent's current policy

        next_state, reward = self.env.step(self.fig, curr_action, display = False)

        # decrease step size using each state-action pair's repetition numbers
        size_step = 1. / (1. + self.visit_actions[curr_state, curr_action])
        self.visit_actions[curr_state, curr_action] += 1.

        self.val_V[curr_state] = size_step * (reward + self.gamma * self.val_V[next_state]) + (1. - size_step) * self.val_V[curr_state]

    # iterate learn_TD0_step
    def learn_TD0_iterate(self, num_iteration = 10000000, thr_converge = 1e-2, prt = False):
        time_prev = time.time() # record the time when TD-learning started
        sqrt_it = int(np.floor(np.sqrt(num_iteration)))
        print "TD-learning started"
        
        # function used for counting the number of certain value in the given list
        countsame = lambda arr, x: sum(map(lambda y: y==x, arr))
        num_escape = 0

        old_valV_it = np.copy(self.val_V)
        old_policy = np.copy(self.mat_policy)
        for i in range(sqrt_it):

            self.env.init_state()
            hist_states = [self.env.state]

            # learn state values of current policy
            for j in range(sqrt_it):
                # perform an action with TD-learning
                self.learn_TD0_step(num_iteration)

                # if the agent visited the last state more than five times in the last 20 actions, reinitialize the agent's state
                hist_states.append(self.env.state)
                if len(hist_states) > 20 and countsame(hist_states[-20:], hist_states[-1]) > 4:
                    self.env.init_state()
                    num_escape += 1
                    hist_states = [self.env.state]

            # update Q-values with current state values, updating policy afterwards
            for j in range(self.num_actions):
                self.val_Q[:, j] = self.env.mat_transition[j].dot(self.val_V)
            self.update_policy(prt)

            # check convergence based on difference sum of state values
            diff_V = np.sum(np.abs(self.val_V - old_valV_it))
            if diff_V < thr_converge: 
                print "State value table converged at step {}, iteration stopped".format((i+1)*sqrt_it)
                break
            old_valV_it = np.copy(self.val_V)

            # report the learning progress, sqrt_it times in total
            if i % (sqrt_it / 20) == 0:
                print "iteration completed: {} / {}".format(i * sqrt_it, num_iteration)
        
        # update policy according to the learnt Q-values
        self.update_policy(prt)
        
        # reset the parameters used for discounting step size of learning
        self.visit_actions = np.zeros([self.env.num_states, self.num_actions])
        time_next = time.time() # record the time when TD-learning ended
        print "TD-learning done; {} seconds passed, {} times reset for escape".format(time_next - time_prev, num_escape)

    def learn_Q_step(self, iteration):
        curr_state = self.env.state

        # choose an action using epsilon-greedy policy
        thr_random = min(self.epsilon, ((iteration - self.idx_it) / iteration))
        self.idx_it += 1.
        if random.random() < thr_random:
            curr_action = random.choice(self.env.actions)
        else: 
            curr_action = np.argmax(self.val_Q[curr_state, :])

        next_state, reward = self.env.step(self.fig, curr_action, display = False) 

        # decrease step size using each state-action pair's repetition numbers
        size_step = 1. / (1. + self.visit_actions[curr_state, curr_action])
        self.visit_actions[curr_state, curr_action] += 1.

        self.val_Q[curr_state, curr_action] = size_step * (reward + self.gamma * np.max(self.val_Q[next_state, :])) + (1. - size_step) * self.val_Q[curr_state, curr_action]
    
    def learn_Q_iterate(self, num_iteration = 10000000, thr_converge = 1e-2, prt = False):
        time_prev = time.time() # record the time when Q-learning started
        sqrt_it = int(np.floor(np.sqrt(num_iteration))) 
        print "Q-learning started"
        
        # function used for counting the number of certain value in the given list
        countsame = lambda arr, x: sum(map(lambda y: y==x, arr))
        num_escape = 0

        old_valQ_it = np.copy(self.val_Q)
        old_policy = np.copy(self.mat_policy)
        for i in range(sqrt_it):

            self.env.init_state()
            hist_states = [self.env.state]
            for j in range(sqrt_it):
                # perform an action with Q-learning
                self.learn_Q_step(num_iteration) 

                # if the agent visited the last state more than five times in the last 20 actions, reinitialize the agent's state
                hist_states.append(self.env.state)
                if len(hist_states) > 20 and countsame(hist_states[-20:], hist_states[-1]) > 4:
                    self.env.init_state()
                    num_escape += 1
                    hist_states = [self.env.state]

            # check convergence based on difference sum of Q values
            diff_Q = np.sum(np.abs(self.val_Q - old_valQ_it))
            if diff_Q < thr_converge: 
                print "Q value table converged at step {}, iteration stopped".format((i+1)*sqrt_it)
                break
            old_valQ_it = np.copy(self.val_Q)

            # report the learning progress, 20 times in total
            if i % (sqrt_it / 20) == 0:
                print "iteration completed: {} / {}".format(i * sqrt_it, num_iteration)
            
        # update policy according to the learnt Q-values
        self.update_policy(prt)
            
        # reset the parameters used for discounting step size of learning
        self.idx_it = 0.
        self.visit_actions = np.zeros([self.env.num_states, self.num_actions])
        time_next = time.time() # record the time when Q-learning ended
        print "Q-learning done; {} seconds passed, {} times reset for escape".format(time_next - time_prev, num_escape)

    ### interaction with environment ###
    def on_key(self, event):
        dict_key = {'right':3,'up':0,'left':2,'down':1}
        dict_dir = {3: 'right', 0: 'up', 2: 'left', 1:'down'}

        # exit figure and code
        if event.key == 'ctrl+c':
            exit(0)

        # move according to the input key, receiving next state and reward
        elif event.key in dict_key.keys():
            state, reward = self.env.step(self.fig, dict_key[event.key])
            self.acc_reward = reward + self.gamma * self.acc_reward
            print "coord: {}, reward: {: 2.4f}, acc_reward: {: 2.4f} ## direction : {}".format(self.env.coord, reward, self.acc_reward, event.key)
            
        # randomly reset the agent's state
        elif event.key == 'i':
            self.env.init_state()
            print "coord: {}".format(self.env.coord)
            self.fig.canvas.draw()
        
        # perform the action with the largest Q-value of current state
        elif event.key == 'a':
            direction = dict_dir[self.vec_policy[self.env.state]]
            curr_action = self.vec_policy[self.env.state]
            state, reward = self.env.step(self.fig, curr_action)
            self.acc_reward = reward + self.gamma * self.acc_reward
            print "coord: {}, reward: {: 2.4f}, acc_reward: {: 2.4f} ## action chosen by policy: {}".format(self.env.coord, reward, self.acc_reward, direction)

        # show how the agent moves, according to the policy calculated by current Q-value
        elif event.key == 'z':
            print "moving 100 times with actions chosen by policy..."
            for i in range(100):
                time.sleep(0.5)
                direction = dict_dir[self.vec_policy[self.env.state]]
                curr_action = self.vec_policy[self.env.state]
                state, reward = self.env.step(self.fig, curr_action)
                self.acc_reward = reward + self.gamma * self.acc_reward
        
        # show the agent's Q-value table
        elif event.key == '3':
            print self.val_Q
        # show the agent's current policy
        elif event.key == '4':
            print self.mat_policy()

        # Dynamic Programming
        elif event.key == 'd':
            self.learn_DP()
        # Q-learning
        elif event.key == 'q':
            self.learn_Q_iterate()
            self.fig.canvas.draw()
        # TD-learning
        elif event.key == 't':
            self.learn_TD0_iterate()
            self.fig.canvas.draw()

        # reset all the learnt parameters
        elif event.key == 'r':
            self.init_everything()
            print "agent forgot everything: needs to learn again"
        
        # increase transition probabilities of the intended actions
        elif event.key == '=':
            if self.env.p_correct < 0.99:
                self.env.p_correct += 0.05
                self.env.mat_transition = self.env.init_transmat(self.env.p_correct)
                print self.env.mat_transition
                self.init_everything()
                print "Transition matrix reinitialized with p_correct: {}\n agent forgot everything;needs to learn again".format(self.env.p_correct)
        # decrease transition probabilities of the intended actions
        elif event.key == '-':
            if self.env.p_correct > 0.05:
                self.env.p_correct -= 0.05
                self.env.mat_transition = self.env.init_transmat(self.env.p_correct)
                print self.env.mat_transition
                self.init_everything()
                print "Transition matrix reinitialized with p_correct: {}\n agent forgot everything;needs to learn again".format(self.env.p_correct)
        # increase gamma value
        elif event.key == '0':
            if self.idx_gamma < (len(self.gammalist) - 1):
                self.idx_gamma += 1
                self.gamma = self.gammalist[self.idx_gamma]
            self.init_everything()
            print "gamma changed to {}".format(self.gamma)
        # decrease gamma value
        elif event.key == '9':
            if self.idx_gamma > 0:
                self.idx_gamma -= 1
                self.gamma = self.gammalist[self.idx_gamma]
            self.init_everything()
            print "gamma changed to {}".format(self.gamma)
        
        # show the optimal action of current action
        elif event.key == '1':
            direction = dict_dir[self.vec_policy[self.env.state]]
            print "learnt optimal action at state {} : {}".format(self.env.state, direction)

        # show the environment spec and parameter setting
        elif event.key == '2':
            print "\n### environment & parameter specs ###"
            print "num. of states: {}\nnum.of states * states: {}\nnum. of actions: {}".format(self.env.num_states, self.env.num_states * self.env.num_states, self.env.num_actions)
            print "p_correct(biggest transition probability): {}\ngamma value: {}\n".format(self.env.p_correct, self.gamma)
            
if __name__ == '__main__':
    agent = RL_agent() # initialize the agent
    print "\n### environment & parameter specs ###"
    print "num. of states: {}\nnum.of states * states: {}\nnum. of actions: {}".format(agent.env.num_states, agent.env.num_states * agent.env.num_states, agent.env.num_actions)
    print "p_correct(biggest transition probability): {}\ngamma value: {}\n".format(agent.env.p_correct, agent.gamma)
    print "rewards of parking: {}".format(agent.env.rewards)
    plt.show() # show the figure
    
