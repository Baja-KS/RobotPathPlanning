#!/usr/bin/env python
# coding: utf-8

# In[121]:


import numpy as np


# In[122]:


class Env:
    def __init__(self, map_height, map_width, max_num_obstacles):
        self.map_height = map_height
        self.map_width = map_width
        self.max_num_obstacles = max_num_obstacles
        self.map = np.ones((map_height, map_width), dtype='int')
        self.map[0:2, :] = 0
        self.map[:, 0:2] = 0
        self.map[map_height-2:, :] = 0
        self.map[:, map_width - 2:] = 0
        
        self.map_copy = np.array(self.map)
        
        print(self.map)
        
        self.init_env()
        
        self.agent_pos = [0, 0]   
        self.target_pos = [map_height - 2, map_width - 2]
    
    def reset_env(self):
        self.map = np.array(self.map_copy)
        self.init_env()
        self.init_agent_pos()
        self.init_target_pos()
        
    def init_env(self):
        for _ in range(np.random.randint(self.max_num_obstacles)):
            obstacle_pos = [0, 0]
            while self.map[obstacle_pos[0], obstacle_pos[1]] == 0:
                obstacle_pos[0] = np.random.randint(2, self.map_height - 2)
                obstacle_pos[1] = np.random.randint(2, self.map_width - 2)
            self.map[obstacle_pos[0], obstacle_pos[1]] = 0
    
    def init_agent_pos(self):
        self.agent_pos[0] = np.random.randint(2, self.map_height - 2)
        self.agent_pos[1] = np.random.randint(2, self.map_width - 2)
        
        while(self.map[self.agent_pos[0], self.agent_pos[1]] == 0):
            self.agent_pos[0] = np.random.randint(2, self.map_height - 2)
            self.agent_pos[1] = np.random.randint(2, self.map_width - 2)
    
    def init_target_pos(self):
        self.target_pos[0] = np.random.randint(2, self.map_height - 2)
        self.target_pos[1] = np.random.randint(2, self.map_width - 2)
        
        while(self.map[self.target_pos[0], self.target_pos[1]] == 0 or               self.target_pos[0] == self.agent_pos[0] and self.target_pos[1] == self.agent_pos[1]):
            self.target_pos[0] = np.random.randint(2, self.map_height - 2)
            self.target_pos[1] = np.random.randint(2, self.map_width - 2)
            
    def get_view(self):
        view = np.zeros(8, dtype='int').tolist()

        view[0:3] = self.map[self.agent_pos[0] - 1][self.agent_pos[1] - 1:self.agent_pos[1] + 2]

        view[3] = self.map[self.agent_pos[0]][self.agent_pos[1] - 1]
        view[4] = self.map[self.agent_pos[0]][self.agent_pos[1] + 1]

        view[5:8] = self.map[self.agent_pos[0] + 1][self.agent_pos[1] - 1:self.agent_pos[1] + 2]

        return view
    
    def is_terminal_state(self):
        if self.map[self.agent_pos[0], self.agent_pos[1]] == 0:
            return True
        if self.target_pos[0] == self.agent_pos[0] and self.target_pos[1] == self.agent_pos[1]:
            return True
        
        return False
    
    def step(self, action, prev_action):
        reward = 0
        done = False
        new_view = env.get_view()
        looping = False
        
        if action == 0:
            self.agent_pos[0] -= 1
            if prev_action == 2:
                looping = True
                
        elif action == 1:
            self.agent_pos[1] += 1
            if prev_action == 3:
                looping = True
                
        elif action == 2:
            self.agent_pos[0] += 1
            if prev_action == 0:
                looping = True
                
        elif action == 3:
            self.agent_pos[1] -= 1
            if prev_action == 1:
                looping = True
        
        reward -= 1
        
        if looping:
            reward - 1000
        
        if self.map[self.agent_pos[0], self.agent_pos[1]] == 0:
            reward -= 1000
            done = True
            
        else:
            new_view = env.get_view()
        
        if self.target_pos[0] == self.agent_pos[0] and self.target_pos[1] == self.agent_pos[1]:
            reward += 1000
            done = True
        
        return new_view, self.agent_pos, reward, done


# In[123]:


class Agent:
    def __init__(self, map_height, map_width, eps, eps_decay, min_eps, lr, lr_decay, gamma):
        self.map_height = map_height
        self.map_width = map_width
        self.eps = eps
        self.min_eps = min_eps
        self.eps_decay = eps_decay
        self.lr = lr
        self.lr_decay = lr_decay
        self.gamma = gamma
        
        self.actions = ['u', 'r', 'd', 'l']
        
        dims = []
        for i in range(8):
            dims.append(2)
        dims.append(self.map_height)
        dims.append(self.map_width)
        dims.append(self.map_height)
        dims.append(self.map_width)
        dims.append(len(self.actions) + 1)
        dims.append(len(self.actions))
        
        self.q_values = np.zeros(dims)
        
        self.q_values[:, 0, :, :, :, :, :, :, :, :, :, :, :, 0] = -10000
        self.q_values[:, :, :, :, 0, :, :, :, :, :, :, :, :, 1] = -10000
        self.q_values[:, :, :, :, :, :, 0, :, :, :, :, :, :, 2] = -10000
        self.q_values[:, :, :, 0, :, :, :, :, :, :, :, :, :, 3] = -10000
        
        self.q_values[:, :, :, :, :, :, :, :, :, :, :, :, 0, 2] = -10000
        self.q_values[:, :, :, :, :, :, :, :, :, :, :, :, 1, 3] = -10000
        self.q_values[:, :, :, :, :, :, :, :, :, :, :, :, 2, 0] = -10000
        self.q_values[:, :, :, :, :, :, :, :, :, :, :, :, 3, 1] = -10000
        
    def act(self, view, row_index, column_index, target_index_x, target_index_y, prev_action):
        if self.eps > np.random.uniform():
            legal_actions = []
            action = np.random.randint(len(self.actions))
        
            if view[1] == 1:
                legal_actions.append(0)
            if view[3] == 1:
                legal_actions.append(3)
            if view[4] == 1:
                legal_actions.append(1)
            if view[6] == 1:
                legal_actions.append(2)
        
            n = len(legal_actions)
            if n != 0:
                action = legal_actions[np.random.randint(n)]
        
            return action
        
        return np.argmax(self.q_values[tuple(view + [row_index, column_index, target_index_x, target_index_y, prev_action])])


# In[124]:


env = Env(13, 13, 16)


# In[125]:


agent = Agent(env.map_height, env.map_width, 1, 0.99999, 0.1, 0.1, 1, 0.99)


# In[126]:


num_episodes = 500000
max_steps = 50
stop_exploring_after = 500000


# In[127]:


#TRAINING
for episode in range(num_episodes):
    env.reset_env()
    
    total_reward = 0
    
    prev_action = 4
    
    if stop_exploring_after == episode:
        agent.eps = agent.min_eps
    
    for step in range(1, max_steps + 1):          
        view = env.get_view()
        agent_pos = list(env.agent_pos)
        target_pos = list(env.target_pos)
        
        
        action = agent.act(view, agent_pos[0], agent_pos[1],
                           target_pos[0], target_pos[1], prev_action)
        
        old_view = list(view)
        old_agent_pos = list(agent_pos)
        old_target_pos = list(target_pos)
        old_prev_action = prev_action
        
        view, agent_pos, reward, done = env.step(action, prev_action)
        
        total_reward += reward
        
        old_q_value = agent.q_values[
                    tuple(old_view + [old_agent_pos[0], old_agent_pos[1],
                                      old_target_pos[0], old_target_pos[1], old_prev_action, action])]

        prev_action = action
        
        temporal_difference = reward + agent.gamma * np.max(
                agent.q_values[tuple(view + [agent_pos[0], agent_pos[1],
                                            target_pos[0], target_pos[1], prev_action])]) - old_q_value

        new_q_value = old_q_value + agent.lr * temporal_difference

        agent.q_values[tuple(
                    old_view + [old_agent_pos[0], old_agent_pos[1],
                                old_target_pos[0], old_target_pos[1], old_prev_action, action])] = new_q_value
        
        if done:
            break
            
    print('e: ' + str(episode) + ' rew: ' + str(total_reward) + ' eps: ' + str(agent.eps))
    if agent.eps > agent.min_eps:
        agent.eps *= agent.eps_decay
    agent.lr *= agent.lr_decay


# In[164]:


#TESTING
test_episodes = 10000
agent.eps = 0.15

tests_passed = 0

env.map = np.array(env.map_copy)
env.init_env()

print(env.map)

for episode in range(test_episodes):
    
    env.init_agent_pos()
    env.init_target_pos()
    
    prev_action = 4
    target_pos = env.target_pos
    
    for step in range(1, max_steps + 1):
        view = env.get_view()
        agent_pos = list(env.agent_pos)
        target_pos = list(env.target_pos)
        
        
        action = agent.act(view, agent_pos[0], agent_pos[1],
                           target_pos[0], target_pos[1], prev_action)
        
        view, agent_pos, reward, done = env.step(action, prev_action)
        
        prev_action = action
        
        if done:
            if agent_pos[0] == target_pos[0] and agent_pos[1] == target_pos[1]:
                tests_passed += 1
            break

print('Success rate: ' + str(tests_passed * 100 / test_episodes) + '%')

