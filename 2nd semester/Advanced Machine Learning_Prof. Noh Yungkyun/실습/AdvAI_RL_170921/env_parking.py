import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

class Parking(object):

    def __init__(self, fig, size_board=(8,8), rewards={(0,4): -1, (1,4): -1, (2,4): -1, (2,7):1},
                            base_reward=0, obstacles=[], pCorrect = 0.99):
        
        self.size_board = size_board
        self.rewards = rewards

        self.base_reward = base_reward # given for every (s,a) unless defined in rewards
        self.obstacles = obstacles
        self.num_car_angles = 8 # 8 directions
        self.num_dw_angles = 3 # left, straight, right
        self.car_len = 3

        num_xs, num_ys = size_board
        self.num_states = (num_xs-2)*(num_ys-2)*self.num_car_angles*self.num_dw_angles
        self.num_actions = 4 # drive, reverse, dw_left, dw_right
        self.actions = range(self.num_actions)

        self.state2coord = [(x,y,ca,dwa) for x in xrange(1,num_xs-1) for y in xrange(1,num_ys-1)
                    for ca in xrange(self.num_car_angles) for dwa in xrange(-1,2)] # hardcoded
        self.coord2state = {c:s for s,c in enumerate(self.state2coord)}
        toint = lambda x: int(np.ceil(np.abs(x))*(1 if x>=0 else -1))
        self.ca2diff = [(toint(e.real), toint(e.imag)) for e in [((1+1j)/np.sqrt(2))**i for i in xrange(8)]]
        self.ca2degree = [45*i for i in xrange(8)]

        self.p_correct = pCorrect # it is good to set 1-p_correct is divisible by 6.
        self.mat_rewards = np.zeros((self.num_states, self.num_actions))


        for i in xrange(self.num_states):
            
            temp_coord = self.state2coord[i]
            def check_overlap(ca):
                if ca in [1,3,5,7]:
                    return [(0,0)]+[self.ca2diff[c%8] for c in [ca, ca+1, ca-1, ca+4, ca+5, ca+3]]
                else:
                    return [(0,0)]+[self.ca2diff[c%8] for c in [ca, ca+4]]

            for c in check_overlap(temp_coord[2]):
                try:
                    self.mat_rewards[i, :] += self.rewards[(temp_coord[0]+c[0], temp_coord[1]+c[1])]
                except:
                    self.mat_rewards[i, :] += self.base_reward
        
        self.mat_transition = self.init_transmat(self.p_correct)
        # self.state = np.random.randint(self.num_states)
        self.state = 1
        self.coord = self.state2coord[self.state]

        ax = fig.add_subplot(111, aspect='equal')
        self.ax = ax
        ax.vlines(np.arange(num_xs), 0, num_ys)
        ax.hlines(np.arange(num_ys), 0, num_xs)
        
        for at,reward in self.rewards.iteritems():
            color = 'green' if reward > 0 else 'red'
            reward_patch = patches.Rectangle((at[0], at[1]), 1, 1, facecolor=color)
            self.ax.add_patch(reward_patch)
            print "reward {} assigned at {}".format(reward, at)

        self.agent_patch = patches.Rectangle((self.coord[0]-.9,self.coord[1]+.1), 2.8, .8, facecolor='yellow') 
        agent_tf = mpl.transforms.Affine2D().rotate_deg_around(self.coord[0]+.5, self.coord[1]+.5, 45*self.coord[2]) + ax.transData
        self.agent_patch.set_transform(agent_tf)

        self.wheel_patch = patches.Rectangle((self.coord[0]+1.2, self.coord[1]+.4), .6, .2, facecolor='black')
        wheel_tf = mpl.transforms.Affine2D().rotate_deg_around(self.coord[0]+1.5, self.coord[1]+.5, 45*self.coord[3]) + ax.transData
        self.wheel_patch.set_transform(wheel_tf)

        self.ax.add_patch(self.agent_patch)
        self.ax.add_patch(self.wheel_patch)
   
    def init_transmat(self, p_correct = 0.94):
        mat_transition = list()    
        list_check = [(1, 1), (1, 6), (6, 1), (6, 6)]
        for a in xrange(self.num_actions):
            mat_transition.append(np.zeros((self.num_states, self.num_states)))

        def yield_possible_transitions(s):
            for a in xrange(self.num_actions):
                coord = self.try_action(s,a)
                try:
                    s_ = self.coord2state[coord]
                    yield (a,s_)
                except KeyError:
                    continue

        for s in xrange(self.num_states):
            pos_trans = list(yield_possible_transitions(s))
            num_pos_trans = len(pos_trans)

            for a in xrange(self.num_actions):
                if a not in [fst for fst,_ in pos_trans]:
                    p_deviate = (1.-p_correct) / (num_pos_trans + 1.)
                    mat_transition[a][s,s] = p_correct + p_deviate
                else:
                    if num_pos_trans == 1:
                        p_deviate = (1.-p_correct) / 2
                        mat_transition[a][s,s] = p_deviate
                    else:
                        p_deviate = (1.-p_correct) / num_pos_trans

                for trans,s_ in pos_trans:
                    mat_transition[a][s,s_] = p_correct + p_deviate if a==trans else p_deviate

        ## setting transition probabilities for states with rewards ##
        for s in xrange(self.num_states):
            if self.mat_rewards[s, 0] != 0:
                for a in xrange(self.num_actions):
                    for s_ in xrange(self.num_states):
                        mat_transition[a][s,s_] = 1. / self.num_states     
        return mat_transition

    def step(self, fig, action, display = True):
        next_state = np.random.choice(self.num_states,
                                p=self.mat_transition[action][self.state, :])
        reward = self.mat_rewards[self.state, action]
        self.state = next_state
        self.coord = self.state2coord[self.state]
        
        if display:    
            self.agent_patch.set_xy([self.coord[0]-.9, self.coord[1]+.1])
            tf = mpl.transforms.Affine2D().rotate_deg_around(self.coord[0]+.5, self.coord[1]+.5, 45*self.coord[2]) + self.ax.transData
            self.agent_patch.set_transform(tf)

            self.wheel_patch.set_xy([self.coord[0]+1.2, self.coord[1]+.4])
            wheel_tf = mpl.transforms.Affine2D().rotate_deg_around(self.coord[0]+1.5, self.coord[1]+.5, 45*self.coord[3]) + tf
            self.wheel_patch.set_transform(wheel_tf)

            fig.canvas.draw()

        return next_state, reward

    def try_action(self, state, action):
        coord = list(self.state2coord[state])
        if action == 2: # dw left
            coord[3] += 1
        elif action == 3: # dw right
            coord[3] -= 1
        else:
            if action == 0: # drive
                coord[2] += coord[3]; coord[2] %= self.num_car_angles
                diff = self.ca2diff[coord[2]]
                coord[0] += diff[0]
                coord[1] += diff[1]
            elif action == 1: # reverse
                diff = self.ca2diff[coord[2]]
                coord[0] -= diff[0]
                coord[1] -= diff[1]
                coord[2] -= coord[3]; coord[2] %= self.num_car_angles
        return tuple(coord)

    def init_state(self):
        self.state = np.random.choice(self.num_states)
        self.coord = self.state2coord[self.state]
             
        self.agent_patch.set_xy([self.coord[0]-.9, self.coord[1]+.1])
        tf = mpl.transforms.Affine2D().rotate_deg_around(self.coord[0]+.5, self.coord[1]+.5, 45*self.coord[2]) + self.ax.transData
        self.agent_patch.set_transform(tf)

        self.wheel_patch.set_xy([self.coord[0]+1.2, self.coord[1]+.4])
        wheel_tf = mpl.transforms.Affine2D().rotate_deg_around(self.coord[0]+1.5, self.coord[1]+.5, 45*self.coord[3]) + tf
        self.wheel_patch.set_transform(wheel_tf)
