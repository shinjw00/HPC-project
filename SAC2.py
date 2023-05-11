import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer(object):
    
    def __init__(self, max_size = 1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        
    def add(self, state, action, reward, next_state, done):
        ind = self.ptr % self.max_size 
        
        if len(self.storage) < self.max_size:
            self.storage.append((state, action, reward, next_state, done))
        else:
            self.storage[ind] = (state, action, reward, next_state, done)
        
        self.ptr += 1
        
    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size = batch_size)
        batch_states = []
        batch_next_states = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        for i in ind:
            state, action, reward, next_state, done = self.storage[i]
            if isinstance(state, tuple):
                state = state[0]
            batch_states.append(state)
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_next_states.append(next_state)
            batch_dones.append(done)
        # print(type(np.array(batch_states)))    <class 'numpy.ndarray'>
        # print('_________________')
        # print(type(np.array(batch_next_states)))    <class 'numpy.ndarray'>
        # print('_________________')
        # print(type(np.array(batch_actions)))    <class 'numpy.ndarray'>
        # print('_________________')
        # print(type(np.array(batch_rewards)))    <class 'numpy.ndarray'>
        # print('_________________')
        # print(type(np.array(batch_dones)))    <class 'numpy.ndarray'>
        # print('_________________')
        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards), np.array(batch_dones)

class Actor(nn.Module):
    
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, action_dim)
        self.max_action = max_action
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        # print(x.shape)    torch.Size([batch_size, state_dim])
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))    # x는 action이므로 max_action 범위 안에 있어야 한다.
        # print(x.shape)    torch.Size([batch_size, action_dim])
        return x
    
    def sample_log_prob(self, state):
        # print(state)    state는 문제 없음
        mean = self.forward(state)    # mean이 문제인데 parameter update 주석 처리하니까 오류 안난다 -> backward() 호출 시, 그랜디언트 발산 
        # print(mean.shape)    torch.Size([batch_size, action_dim])
        log_std = self.log_std.expand_as(mean)
        # print(log_std.shape)    torch.Size([batch_size, action_dim])
        std = log_std.exp()    # log_std = 1 -> std = e^1
        # print(std.shape)    torch.Size([batch_size, action_dim])
        # print(std)    std는 문제 없음
        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()    # Size (batch_size, action_dim)
        action = torch.tanh(z)
        # print(action.shape)    # torch.Size([batch_size, action_dim])
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        # print(log_prob.shape)    torch.Size([batch_size, action_dim])
        return action, log_prob.sum(dim=1, keepdim=True)
    
class Critic(nn.Module):

  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    # Defining the first Critic neural network
    self.layer_1 = nn.Linear(state_dim + action_dim, 256)
    self.layer_2 = nn.Linear(256, 256)
    self.layer_3 = nn.Linear(256, 1)     # Q는 한개만 ouput으로 나오므로 Q_dim일 필요가 없다. Q_dim = 1 
    # Defining the second Critic neural network
    self.layer_4 = nn.Linear(state_dim + action_dim, 256)
    self.layer_5 = nn.Linear(256, 256)
    self.layer_6 = nn.Linear(256, 1)
  
  def forward(self, x, u):    # x는 input state, u는 action plate
    xu = torch.cat([x, u], 1)    # axis = 1 vertical, axis = 0 horizontal
    # print(xu.shape)   # torch.Size([batch_size, state_dim + action_dim])          
    # Forward-propagation on the first Critic neural network
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1) 
    # print(x1.shape)    torch.Size([batch_size, 1]) -> Q-value 1개 
    # Forward-propagation on the second Critic neural network
    x2 = F.relu(self.layer_4(xu))
    x2 = F.relu(self.layer_5(x2))
    x2 = self.layer_6(x2)
    return x1, x2       
    
  def Q1(self, x, u):    
    xu = torch.cat([x, u], 1)    # axis = 1 vertical, axis = 0 horizontal
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1) 
    return x1


class SAC(object):
    
    def __init__(self, state_dim, acion_dim, max_action, alpha, beta):
        self.actor = Actor(state_dim, acion_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = alpha)
        self.critic = Critic(state_dim, acion_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = beta)
        self.target_critic = Critic(state_dim, acion_dim).to(device)
        self.target_critic_optimizer = optim.Adam(self.critic.parameters(), lr = beta)
        
    def select_action(self, state):
        state = torch.from_numpy((state).reshape(1, -1)).to(device)
        with torch.no_grad():
            return self.actor(state).cpu().numpy().flatten()
        
    def train(self, iterations, replay_buffer, batch_size, alpha_1, discount_factor, policy_freq, tau):
        for it in range(iterations):
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            state = torch.Tensor(batch_states).to(device) 
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)
            
            # Critic training
            next_action, next_log_prob = self.actor.sample_log_prob(next_state)
            # print(next_action.shape)    torch.Size([batch_size, action_dim])
            # print(next_log_prob.shape)    torch.Size([batch_size, action_dim])
            target_Q1, target_Q2 = self.target_critic(next_state, next_action)
            # print(target_Q1.shape)     torch.Size([batch_size, action_dim])
            # print(target_Q2.shape)     torch.Size([batch_size, action_dim])
            target_Q = torch.min(target_Q1, target_Q2) - alpha_1 * next_log_prob
            # print(target_Q.shape)    torch.Size([batch_size, action_dim])
            target_Q = reward.reshape(-1, 1) + (discount_factor * (1 - done.reshape(-1, 1)) * target_Q)
            # print(reward.shape)     torch.Size([batch_size, action_dim])
            # print(done.shape)     torch.Size([batch_size, action_dim])
            #print(target_Q.shape)     torch.Size([batch_size, action_dim])
            
            current_Q1, current_Q2 = self.critic(state, action)
            # print(current_Q1.shape)     torch.Size([batch_size, action_dim])
            # print(current_Q2.shape)     torch.Size([batch_size, action_dim])
            # print(state.shape)    torch.Size([batch_size, state_dim])
            # print(action.shape)    torch.Size([batch_size, action_dim])
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm = 1.0)
            self.critic_optimizer.step()
            
            # Actor training
            if it % policy_freq == 0:
                _, log_prob = self.actor.sample_log_prob(state)
                
                # print(state.shape)   torch.size([batch_size, state_dim])
                actor_loss = -(self.critic.Q1(state, self.actor(state)) - alpha_1 * log_prob).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Target networks update
                for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

env = gym.make("Pendulum-v1", render_mode = 'human')

state_dim = env.observation_space.shape[0]
# print(state_dim)    3
action_dim = env.action_space.shape[0]
# print(action_dim)    1
max_action = float(env.action_space.high[0])
alpha = 0.0005
beta = 0.0005
iterations = 100

policy = SAC(state_dim, action_dim, max_action, alpha, beta)

replay_buffer = ReplayBuffer()

batch_size = 128
alpha_1 = 0.2 
discount_factor = 0.99 
policy_freq = 2
update_freq = 5
tau = 0.05
episode_num = 1
episode_reward = 0
episode_timesteps = 0
start_timesteps = int(1000)
total_timesteps = int(10000)
done = True

episode_reward = 0
episode_timesteps = 0
obs = env.reset() 

timestep = [] 
reward_timestep = []
log_prob_timestep = []

for t in range(total_timesteps):
    print(t)
    # Random action for initial exploration
    if t < start_timesteps:
        action = env.action_space.sample()
    else:
        action = policy.select_action(obs)

    # Perform action and store the transition in the replay buffer
    new_obs, reward, done, _, _ = env.step(action)
    reward_timestep.append(reward)
    timestep.append(t)
    replay_buffer.add(obs, action, reward, new_obs, float(done))
    obs = new_obs

    episode_timesteps += 1

    if t >= start_timesteps and t % update_freq ==0:
        policy.train(iterations, replay_buffer, batch_size, alpha_1, discount_factor, policy_freq, tau)
        
    env.render()
  
plt.plot(timestep, reward_timestep, 'r--')
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.title('Reward over Timesteps')

plt.show()  
print("Training finished.")