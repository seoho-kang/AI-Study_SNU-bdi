import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Maze(object):
    '''
    def __init__(self, fig, size_board=(8,6), num_actions = 4,
                 rewards={(7,5): 1., (7,4): -0.7}, base_reward=0., obstacles=[(5,5), (5,4), (7,2),(6,2),(5,2),(4,2)], pCorrect = 0.94):
    '''
    def __init__(self, fig, size_board=(4,3), num_actions = 4,
                 rewards={(3,2): 1., (3,1): -0.7}, base_reward=0., obstacles=[(1,1)], pCorrect = 0.94):

        self.size_board = size_board
        self.rewards = rewards
        
        self.base_reward = base_reward
        self.obstacles = obstacles
        
        num_xs, num_ys = size_board
        self.num_states = num_xs * num_ys - len(obstacles)
        self.num_actions = num_actions
        self.actions = range(self.num_actions)
        
        self.state2coord = list()
        for x in xrange(num_xs):
            for y in xrange(num_ys):
                if (x, y) not in obstacles: self.state2coord.append((x, y))
        
        self.coord2state = dict()
        for i in xrange(len(self.state2coord)):
            self.coord2state[self.state2coord[i]] = i
        
        self.action2diff = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.p_correct = pCorrect # it is good to set 1-p_correct is divisible by 6.
        
        self.mat_rewards = np.zeros((self.num_states, self.num_actions))
        
        for i in xrange(self.num_states):
            try:
                self.mat_rewards[i, :] = self.rewards[self.state2coord[i]]
                print "State {} gets reward {}".format(i, self.rewards[self.state2coord[i]])
            except:
                self.mat_rewards[i, :] = base_reward
        
        self.mat_transition = self.init_transmat(self.p_correct)
        self.state = 0
        self.coord = self.state2coord[self.state]

        ### figure of the maze ###
        ax = fig.add_subplot(111, aspect='equal')
        self.ax = ax
        ax.vlines(np.arange(num_xs), 0, num_ys)
        ax.hlines(np.arange(num_ys), 0, num_xs)
        for obs in self.obstacles:
            ax.add_patch( patches.Rectangle((obs[0],obs[1]), 1,1, hatch='/', fill=False) )
        for coord,reward in self.rewards.iteritems():
            color = 'green' if reward > 0 else 'red'
            ax.add_patch( patches.Rectangle((coord[0], coord[1]), 1,1, facecolor=color) )
        
        self.agent_patch = patches.Rectangle((self.coord[0]+.1,self.coord[1]+.1), .8,.8, facecolor='yellow') 
        ax.add_patch(self.agent_patch)
        ######
        
        ### drawing arrows ###
        self.arrows = list()
        for st in range(self.num_states):
            cd = self.state2coord[st]
            if cd in self.rewards.keys(): 
                self.arrows.append([])
                continue
            
            dy = 0
            dx = .35
            
            self.arrows.append(patches.FancyArrowPatch((cd[0]+.5, cd[1]+.5),(cd[0]+.5+dx, cd[1]+.5+dy), 
                                                       arrowstyle = '->', mutation_scale = 2.0))
            self.arrows[st].set_arrowstyle("fancy", head_width = 4.0, head_length = 4.0, tail_width = 1.5)
            self.ax.add_patch(self.arrows[st])
        ######
        
    def redraw_arrows(self, fig, vec_policy):
        for st in range(self.num_states):
            cd = self.state2coord[st]
            if cd in self.rewards.keys(): continue
            direction = vec_policy[st]
            if direction % 2 == 0: 
                dy = 0
                if direction > 1: dx = -.35
                else: dx = .35
            else: 
                dx = 0
                if direction > 2: dy = -.35
                else: dy = .35
            self.arrows[st].set_positions((cd[0]+.5, cd[1]+.5),(cd[0]+.5+dx, cd[1]+.5+dy))
        fig.canvas.draw()
        
    def init_transmat(self, p_correct = 0.94):
        mat_transition = list()
        for a in xrange(self.num_actions):
            mat_transition.append(np.zeros((self.num_states, self.num_states)))
        
        def yield_possible_transitions(s):
            x,y = self.state2coord[s]
            for a in xrange(self.num_actions):
                diff = self.action2diff[a]
                try:
                    s_ = self.coord2state[(x+diff[0], y+diff[1])]
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
                    p_deviate = (1.-p_correct) / num_pos_trans

                for trans,s_ in pos_trans:
                    mat_transition[a][s,s_] = p_correct + p_deviate if a==trans else p_deviate
        
        ### setting transition probabilities for states with rewards ###
        for rew in self.rewards.keys():
            s = self.coord2state[rew]
            for a in xrange(self.num_actions):
                for s_ in xrange(self.num_states):
                    mat_transition[a][s,s_] = 1. / self.num_states
        
        return mat_transition

    def init_state(self):
        self.state = np.random.choice(self.num_states)
        self.coord = self.state2coord[self.state]
        self.agent_patch.set_xy([c+.1 for c in self.coord])

    def step(self, fig, action, display = True):
        curr_state = self.state
        self.state = np.random.choice(self.num_states, p=self.mat_transition[action][self.state, :])
        self.coord = self.state2coord[self.state]
        
        try:
            reward = self.mat_rewards[curr_state, action]
        except:
            reward = self.base_reward

        if display:
            self.agent_patch.set_xy([c+.1 for c in self.coord])
            fig.canvas.draw()
        
        return self.state, reward    
