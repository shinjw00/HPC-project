import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer(object):

  def __init__(self, max_size = int(1e6)):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0  # pointer index 쓰기 위해서

  def add(self, state, action, reward, next_state, done):
    data = (state, action, reward, next_state, done)
    if len(self.storage) < self.max_size:
        self.storage.append(data)
    else:
        ind = self.ptr % self.max_size
        self.storage[ind] = data
    
    self.ptr += 1

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind:
        state, action, reward, next_state, done = self.storage[i]
        batch_states.append(state)
        batch_next_states.append(next_state)
        batch_actions.append(action)
        batch_rewards.append(reward)
        batch_dones.append(done)
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)
          
class Actor(nn.Module):

  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, action_dim)
    self.max_action = max_action
    self.log_std = nn.Parameter(torch.zeros(action_dim))  # 로그 표준편차 파라미터 추가
  
  def forward(self, x):    # x는 input state
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = self.max_action * torch.tanh(self.layer_3(x))  # tanh는 -1과 1 사이의 값만 가지는데 이를 원래의 action rang에 맞게 조정하기 위해 max_action 곱해줌 
    return x
  
  def sample_log_prob(self, state):
    mean = self.forward(state)
    log_std = self.log_std.expand_as(mean)
    std = log_std.exp()
    normal = torch.distributions.Normal(mean, std)
    z = normal.sample()
    action = torch.tanh(z)
    log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
    return action, log_prob.sum(dim=1, keepdim=True)

class Critic(nn.Module):

  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    # Defining the first Critic neural network
    self.layer_1 = nn.Linear(state_dim + action_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, 1)  # Q는 한개만 ouput으로 나오므로 Q_dim일 필요가 없다. Q_dim = 1 
    # Defining the second Critic neural network
    self.layer_4 = nn.Linear(state_dim + action_dim, 400)
    self.layer_5 = nn.Linear(400, 300)
    self.layer_6 = nn.Linear(300, 1) 
  
  def forward(self, x, u):    # x는 input state, u는 action plate
    xu = torch.cat([x, u], 1)  # axis = 1 vertical, axis = 0 horizontal
    # Forward-propagation on the first Critic neural network
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1) 
    # Forward-propagation on the second Critic neural network
    x2 = F.relu(self.layer_4(xu))
    x2 = F.relu(self.layer_5(x2))
    x2 = self.layer_6(x2)
    return x1, x2       # 여기까지만 만들고 4개의 다른 객체(critic model 2, critic target 2)를 만들어도 됨.

class SAC(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters())
        self.target_critic = Critic(state_dim, action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha])

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            return self.actor(state).cpu().numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.005, target_entropy=-4.0, policy_freq = 2):
        for it in range(iterations):
            state, next_state, action, reward, done = replay_buffer.sample(batch_size)
            state = torch.Tensor(state).to(device)
            next_state = torch.Tensor(next_state).to(device)
            action = torch.Tensor(action).to(device)
            reward = torch.Tensor(reward).to(device)
            done = torch.Tensor(done).to(device)

            # Critic training
            with torch.no_grad():
                next_action, next_log_prob = self.actor.sample_log_prob(next_state)
                target_Q1, target_Q2 = self.target_critic(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2) - self.log_alpha.exp() * next_log_prob
                target_Q = reward + (1 - done) * discount * target_Q

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor training
            if it % policy_freq == 0:
                new_action, log_prob = self.actor.sample_log_prob(state)
                Q1, Q2 = self.critic(state, new_action)
                Q = torch.min(Q1, Q2)
                actor_loss = (self.log_alpha.exp() * log_prob - Q).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Alpha training
                alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                # Target networks update
                for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    return avg_reward / eval_episodes


# Set the environment
env = gym.make("Pendulum-v1", render_mode ='human')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize SAC agent
sac_agent = SAC(state_dim, action_dim, max_action)

# Initialize Replay Buffer
replay_buffer = ReplayBuffer()

# Set number of training iterations
iterations = 50000
batch_size = 128
start_timesteps = 1000

# Main training loop
timesteps_since_eval = 0
episode_num = 0
episode_reward = 0
episode_timesteps = 0
done = True

for t in range(iterations):
    if done:
        if t != 0:
            print(f"Total T: {t} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward}")
            sac_agent.train(replay_buffer, iterations, batch_size, discount=0.99, tau=0.005, target_entropy=-4.0, policy_freq = 2)
        
        # Evaluate the policy
        if timesteps_since_eval >= start_timesteps:
            timesteps_since_eval %= start_timesteps
            evaluations = evaluate_policy(sac_agent)
            print(f"Evaluation over {start_timesteps} episodes: {evaluations}")
        
        # Reset environment
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
    
    # Select action
    if t < start_timesteps:
        action = env.action_space.sample()
    else:
        action = sac_agent.select_action(np.array(obs))
    
    # Perform action
    new_obs, reward, done, _, _ = env.step(action)
    done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
    
    # Store experience in replay buffer
    replay_buffer.add(obs, action, reward, new_obs, done_bool)
    
    # Update values
    obs = new_obs
    episode_reward += reward
    episode_timesteps += 1
    timesteps_since_eval += 1
print("Training finished.")
env.render()