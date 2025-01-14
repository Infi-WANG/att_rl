import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns 
import rospkg
import random
from env_test import UREnv
from env_chamfered import UREnv_chamfer
from copy import deepcopy
from geometry_msgs.msg import Pose
from torch.distributions import Normal

force_scale = 10.0
torque_scale = 0.8

class ValueNet(nn.Module):
    def __init__(self, n_states, hidden_dim, init_w=3e-3):
        super(ValueNet, self).__init__()
        '''定义值网络
        '''
        self.linear1 = nn.Linear(n_states, hidden_dim) # 输入层
        self.linear2 = nn.Linear(hidden_dim, hidden_dim) # 隐藏层
        self.linear3 = nn.Linear(hidden_dim, hidden_dim) # 隐藏层
        self.linear4 = nn.Linear(hidden_dim, 1)

        self.linear4.weight.data.uniform_(-init_w, init_w) # 初始化权重
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x    
    
class SoftQNet(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3):
        super(SoftQNet, self).__init__()
        '''定义Q网络，n_states, n_actions, hidden_dim, init_w分别为状态维度、动作维度隐藏层维度和初始化权重
        '''
        self.linear1 = nn.Linear(n_states + n_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
    
class PolicyNet(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNet, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, n_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_dim, n_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        ## 计算动作
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        ## 计算动作概率
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob, z, mean, log_std
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        action = action.detach().cpu().numpy()
        return action[0]

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.positon = 0

    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.positon] = (state, action, reward, next_state, done)
        self.positon = (self.positon + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # 随机采出小批量转移
        state, action, reward, next_state, done =  zip(*batch) # 解压成状态，动作等
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
    
class SAC:
    def __init__(self, cfg) -> None:
        self.n_states = cfg.n_states
        self.n_actions = cfg.n_actions
        self.batch_size = cfg.batch_size
        self.memory = ReplayBuffer(cfg.capacity)
        self.device = cfg.device
        self.action_space = cfg.action_space
        self.value_net, self.target_value_net, self.soft_q_net, self.policy_net = load_model()
        # self.value_net = ValueNet(self.n_states, cfg.hidden_dim).to(self.device)
        # self.target_value_net = ValueNet(self.n_states, cfg.hidden_dim).to(self.device)
        # self.soft_q_net = SoftQNet(self.n_states, self.n_actions, cfg.hidden_dim).to(self.device)
        # self.policy_net = PolicyNet(self.n_states, self.n_actions, cfg.hidden_dim).to(self.device)
        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=cfg.value_lr)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=cfg.soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.policy_lr)
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)
        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()
    def update(self, gamma=0.99, mean_lambda=1e-3,
        std_lambda=1e-3,
        z_lambda=0.0,
        soft_tau=1e-2,
        ):
        if len(self.memory) < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        expected_q_value = self.soft_q_net(state, action) #计算t时刻的状态-动作Q值
        expected_value = self.value_net(state) #计算t时刻的状态值
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state) #计算t时刻的动作、动作似然概率、正态分布抽样、分布均值和标准差

        target_value = self.target_value_net(next_state) #计算t+1时刻的状态值
        next_q_value = reward + (1 - done) * gamma * target_value # 时序差分计算t+1时刻的Q值
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach()) #计算q网路的损失函数

        expected_new_q_value = self.soft_q_net(state, new_action) #计算t时刻动作对应的q值
        next_value = expected_new_q_value - log_prob # 计算t时刻的v值
        value_loss = self.value_criterion(expected_value, next_value.detach()) #计算值网络损失函数
        
        ## 计算策略损失
        log_prob_target = expected_new_q_value - expected_value 
        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

        ## 计算reparameterization参数损失
        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss = std_lambda * log_std.pow(2).mean()
        z_loss = z_lambda * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        ## 更新目标值网络参数
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

class Insert_SAC:
    def __init__(self, cfg) -> None:
        self.n_states = cfg.n_inserts
        self.n_actions = cfg.n_actions
        self.batch_size = cfg.insert_batch_size
        self.memory = ReplayBuffer(cfg.capacity)
        self.device = cfg.device
        self.action_space = cfg.action_space
        self.value_net, self.target_value_net, self.soft_q_net, self.policy_net = load_insert_model()
        # self.value_net = ValueNet(self.n_states, cfg.hidden_dim).to(self.device)
        # self.target_value_net = ValueNet(self.n_states, cfg.hidden_dim).to(self.device)
        # self.soft_q_net = SoftQNet(self.n_states, self.n_actions, cfg.hidden_dim).to(self.device)
        # self.policy_net = PolicyNet(self.n_states, self.n_actions, cfg.hidden_dim).to(self.device)
        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=cfg.insert_value_lr)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=cfg.insert_soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.insert_policy_lr)
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)
        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()
    def update(self, gamma=0.99, mean_lambda=1e-3,
        std_lambda=1e-3,
        z_lambda=0.0,
        soft_tau=1e-2,
        ):
        if len(self.memory) < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        expected_q_value = self.soft_q_net(state, action) #计算t时刻的状态-动作Q值
        expected_value = self.value_net(state) #计算t时刻的状态值
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state) #计算t时刻的动作、动作似然概率、正态分布抽样、分布均值和标准差

        target_value = self.target_value_net(next_state) #计算t+1时刻的状态值
        next_q_value = reward + (1 - done) * gamma * target_value # 时序差分计算t+1时刻的Q值
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach()) #计算q网路的损失函数

        expected_new_q_value = self.soft_q_net(state, new_action) #计算t时刻动作对应的q值
        next_value = expected_new_q_value - log_prob # 计算t时刻的v值
        value_loss = self.value_criterion(expected_value, next_value.detach()) #计算值网络损失函数
        
        ## 计算策略损失
        log_prob_target = expected_new_q_value - expected_value 
        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

        ## 计算reparameterization参数损失
        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss = std_lambda * log_std.pow(2).mean()
        z_loss = z_lambda * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        ## 更新目标值网络参数
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

def train(cfg, env, agent, target):
    print('Start training!')
    rewards_in_round = []
    ma_rewards_in_round = []
    for i_ep in range(cfg.train_eps):
        state = env.reset(target)
        reward_in_round = 0
        for step in range(cfg.max_steps):
            action = agent.policy_net.get_action(state)  # 抽样动作

            low  = -1*force_scale
            high = force_scale
            action = low + (action + 1.0) * 0.5 * (high - low)
            action = np.clip(action, low, high)

            next_state, reward, done = env.step(action)  # 更新环境，返回transitions
            reward = reward*cfg.reward_scale
            agent.memory.push(state, action/force_scale, reward, next_state, done)  # 保存transition
            agent.update(cfg.gamma, cfg.mean_lambda, cfg.std_lambda, cfg.z_lambda, cfg.soft_tau)
            state = next_state
            reward_in_round += reward
            if done or step==cfg.max_steps-1:
                rewards_in_round.append(reward_in_round)
                if ma_rewards_in_round:
                    ma_rewards_in_round.append(0.9*ma_rewards_in_round[-1]+0.1*reward_in_round) 
                else:
                    ma_rewards_in_round.append(reward_in_round)
                plot_rewards(rewards_in_round, ma_rewards_in_round)
                if (i_ep+1) % 50 == 0:
                    save_model(agent.value_net, agent.target_value_net, agent.soft_q_net, agent.policy_net, i_ep)
                    make_file(rewards_in_round, ma_rewards_in_round)
                break

def train_insert(cfg, env, agent, insert_agent, target):
    print('Start training!')
    insert_rewards_in_round = []
    insert_ma_rewards_in_round = []
    for i_ep in range(cfg.train_eps):
        state = env.reset(target)
        reward_in_round = 0
        for step in range(cfg.max_steps):
            action = agent.policy_net.get_action(state)  # 抽样动作

            low  = -1*force_scale
            high = force_scale
            action = low + (action + 1.0) * 0.5 * (high - low)
            action = np.clip(action, low, high)

            next_state, reward, done = env.step(action)  # 更新环境，返回transitions
            reward = reward*cfg.reward_scale
            state = next_state
            reward_in_round += reward

            if done and reward_in_round > -1000:
                insert_state = env.insert()
                insert_reward_in_round = 0
                for i in range(cfg.max_insert_steps):
                    insert_action = insert_agent.policy_net.get_action(insert_state)

                    low  = -1*torque_scale
                    high = 1*torque_scale
                    insert_action = low + (insert_action + 1.0) * 0.5 * (high - low)
                    insert_action = np.clip(insert_action, low, high)

                    insert_next_state, insert_reward, insert_done = env.insert_step(insert_action)
                    insert_reward = insert_reward*cfg.reward_scale
                    insert_agent.memory.push(insert_state, insert_action/torque_scale, insert_reward, insert_next_state, insert_done)
                    insert_agent.update(cfg.gamma, cfg.mean_lambda, cfg.std_lambda, cfg.z_lambda, cfg.soft_tau)
                    insert_state = insert_next_state
                    insert_reward_in_round += insert_reward
                    
                    if insert_done or i==cfg.max_insert_steps-1:
                        insert_rewards_in_round.append(insert_reward_in_round)
                        if insert_ma_rewards_in_round:
                            insert_ma_rewards_in_round.append(0.9*insert_ma_rewards_in_round[-1]+0.1*insert_reward_in_round)
                        else:
                            insert_ma_rewards_in_round.append(insert_reward_in_round)
                        plot_insert_rewards(insert_rewards_in_round, insert_ma_rewards_in_round)
                        if len(insert_rewards_in_round) % 50 == 0:
                            save_insert_model(insert_agent.value_net, insert_agent.target_value_net, insert_agent.soft_q_net, insert_agent.policy_net, i_ep)
                            make_insert_file(insert_rewards_in_round, insert_ma_rewards_in_round)
                        break

            if done or step==cfg.max_steps-1:
                break

def test(cfg, env, agent, insert_agent, target):
    print("开始测试！")
    steps = []
    s_steps = []
    i_steps = []
    insert_rewards_in_round = []
    insert_ma_rewards_in_round = []
    rewards_in_round = []
    ma_rewards_in_round = []
    for i_ep in range(cfg.test_eps):
        state = env.reset(target)
        reward_in_round = 0
        for step in range(cfg.max_steps):
            action = agent.policy_net.get_action(state)  # 抽样动作

            low  = -1*force_scale
            high = force_scale
            action = low + (action + 1.0) * 0.5 * (high - low)
            action = np.clip(action, low, high)

            next_state, reward, done = env.step(action)  # 更新环境，返回transitions
            reward = reward*cfg.reward_scale
            state = next_state
            reward_in_round += reward

            if done and reward_in_round > -1000:
                insert_state = env.insert()
                insert_reward_in_round = 0
                for i in range(cfg.max_insert_steps):
                    insert_action = insert_agent.policy_net.get_action(insert_state)

                    low  = -1*torque_scale
                    high = 1*torque_scale
                    insert_action = low + (insert_action + 1.0) * 0.5 * (high - low)
                    insert_action = np.clip(insert_action, low, high)

                    insert_next_state, insert_reward, insert_done = env.insert_step(insert_action)
                    insert_reward = insert_reward*cfg.reward_scale
                    insert_state = insert_next_state
                    insert_reward_in_round += insert_reward
                    
                    if insert_done or i==cfg.max_insert_steps-1:
                        insert_rewards_in_round.append(insert_reward_in_round)
                        if insert_ma_rewards_in_round:
                            insert_ma_rewards_in_round.append(0.9*insert_ma_rewards_in_round[-1]+0.1*insert_reward_in_round)
                        else:
                            insert_ma_rewards_in_round.append(insert_reward_in_round)
                        make_insert_file(insert_rewards_in_round, insert_ma_rewards_in_round)

                        if done and insert_done:
                            steps.append(step+i+2)
                            make_test_file(steps)
                            i_steps.append(i+1)
                            insert_step(i_steps)
                            s_steps.append(step+1)
                            search_step(s_steps)
                        
                        break

            if done or step==cfg.max_steps-1:
                rewards_in_round.append(reward_in_round)
                if ma_rewards_in_round:
                    ma_rewards_in_round.append(0.9*ma_rewards_in_round[-1]+0.1*reward_in_round) 
                else:
                    ma_rewards_in_round.append(reward_in_round)
                if (i_ep+1) % 50 == 0:
                    make_file(rewards_in_round, ma_rewards_in_round)
                break

    print("完成测试！")

def env_agent_config(cfg):
    # env = UREnv() # 创建环境
    env = UREnv_chamfer()
    n_states = env.observation_space.shape[0]
    n_inserts = env.insert_space.shape[0]
    n_actions = env.action_space.shape[0]
    print(f"状态空间维度：{n_states}，动作空间维度：{n_actions}")
    # 更新n_states和n_actions到cfg参数中
    setattr(cfg, 'n_states', n_states)
    setattr(cfg, 'n_actions', n_actions) 
    setattr(cfg, 'n_inserts', n_inserts) 
    setattr(cfg, 'action_space', env.action_space) 
    agent = SAC(cfg)
    insert_agent = Insert_SAC(cfg)
    return env, agent, insert_agent

def make_test_file(steps):
    step = deepcopy(steps)
    s2 = '\n'
    for i in range(len(step)):
        step[i]=str(step[i])
    f2=open("steps.txt", "w")
    f2.write(s2.join(step))
    f2.close()

def search_step(steps):
    step = deepcopy(steps)
    s2 = '\n'
    for i in range(len(step)):
        step[i]=str(step[i])
    f2=open("search_steps.txt", "w")
    f2.write(s2.join(step))
    f2.close()

def insert_step(steps):
    step = deepcopy(steps)
    s2 = '\n'
    for i in range(len(step)):
        step[i]=str(step[i])
    f2=open("insert_step.txt", "w")
    f2.write(s2.join(step))
    f2.close()

def make_file(rewards, ma_rewards):
    rd = deepcopy(rewards)
    s1 = '\n'
    for i in range(len(rd)):
        rd[i]=str(rd[i])
    f1=open("rewards.txt", "w")
    f1.write(s1.join(rd))
    f1.close()

    mard = deepcopy(ma_rewards)
    s2 = '\n'
    for i in range(len(mard)):
        mard[i]=str(mard[i])
    f2=open("ma_rewards.txt", "w")
    f2.write(s2.join(mard))
    f2.close()

def make_insert_file(rewards, ma_rewards):
    rd = deepcopy(rewards)
    s1 = '\n'
    for i in range(len(rd)):
        rd[i]=str(rd[i])
    f1=open("insert_rewards.txt", "w")
    f1.write(s1.join(rd))
    f1.close()

    mard = deepcopy(ma_rewards)
    s2 = '\n'
    for i in range(len(mard)):
        mard[i]=str(mard[i])
    f2=open("insert_ma_rewards.txt", "w")
    f2.write(s2.join(mard))
    f2.close()

def plot_rewards(rewards, ma_rewards):
    sns.set()
    plt.ion()
    plt.clf()
    plt.figure(1)
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    plt.pause(0.1)
    plt.ioff()

def plot_insert_rewards(rewards, ma_rewards):
    sns.set()
    plt.ion()
    plt.clf()
    plt.figure(1)
    plt.xlabel('insert epsiodes')
    plt.plot(rewards, label='insert rewards')
    plt.plot(ma_rewards, label='insert ma rewards')
    plt.legend()
    plt.pause(0.1)
    plt.ioff()

def save_model(vn, tvn, sqn, pn, epoch):
    rospack = rospkg.RosPack()
    PATH = '/home/infi/rl_ws/src/admitance_ur/compliance_control/admittance_controller'
    vPATH = PATH + '/model/' + time.strftime("%Y-%m-%d %H:%M", time.localtime()) +' vn' + str(epoch) + '.pt'
    tvPATH = PATH + '/model/' + time.strftime("%Y-%m-%d %H:%M", time.localtime()) +' tvn' + str(epoch) + '.pt'
    sqPATH = PATH + '/model/' + time.strftime("%Y-%m-%d %H:%M", time.localtime()) +' sqn' + str(epoch) + '.pt'
    pPATH = PATH + '/model/' + time.strftime("%Y-%m-%d %H:%M", time.localtime()) +' pn' + str(epoch) + '.pt'
    torch.save(vn.state_dict(), vPATH)
    torch.save(tvn.state_dict(), tvPATH)
    torch.save(sqn.state_dict(), sqPATH)
    torch.save(pn.state_dict(), pPATH)

def save_insert_model(vn, tvn, sqn, pn, epoch):
    rospack = rospkg.RosPack()
    PATH = '/home/infi/rl_ws/src/admitance_ur/compliance_control/admittance_controller'
    vPATH = PATH + '/model/' + 'insert' +' vn' + str(epoch) + '.pt'
    tvPATH = PATH + '/model/' + 'insert' +' tvn' + str(epoch) + '.pt'
    sqPATH = PATH + '/model/' + 'insert' +' sqn' + str(epoch) + '.pt'
    pPATH = PATH + '/model/' + 'insert' +' pn' + str(epoch) + '.pt'
    torch.save(vn.state_dict(), vPATH)
    torch.save(tvn.state_dict(), tvPATH)
    torch.save(sqn.state_dict(), sqPATH)
    torch.save(pn.state_dict(), pPATH)

def load_model():
    rospack = rospkg.RosPack()
    PATH = '/home/infi/rl_ws/src/admitance_ur/compliance_control/admittance_controller'
    vn = PATH + '/model/2023-08-30/' + 'vn2799.pt'
    tvn = PATH + '/model/2023-08-30/' + 'tvn2799.pt'
    sqn = PATH + '/model/2023-08-30/' + 'sqn2799.pt'
    pn = PATH + '/model/2023-08-30/' + 'pn2799.pt'
    # vn = PATH + '/model/2023-09-27/' + 'vn5499.pt'
    # tvn = PATH + '/model/2023-09-27/' + 'tvn5499.pt'
    # sqn = PATH + '/model/2023-09-27/' + 'sqn5499.pt'
    # pn = PATH + '/model/2023-09-27/' + 'pn5499.pt'
    vnmodel = ValueNet(10, 256)
    vnmodel.load_state_dict(torch.load(vn))
    vnmodel.eval()
    tvnmodel = ValueNet(10, 256)
    tvnmodel.load_state_dict(torch.load(tvn))
    tvnmodel.eval()
    sqnmodel = SoftQNet(10, 2, 256)
    sqnmodel.load_state_dict(torch.load(sqn))
    sqnmodel.eval()
    pnmodel = PolicyNet(10, 2, 256)
    pnmodel.load_state_dict(torch.load(pn))
    pnmodel.eval()
    return vnmodel, tvnmodel, sqnmodel, pnmodel

def load_insert_model():
    rospack = rospkg.RosPack()
    PATH = '/home/infi/rl_ws/src/admitance_ur/compliance_control/admittance_controller'
    vn = PATH + '/model/2023-09-12/' + 'vn1749.pt'
    tvn = PATH + '/model/2023-09-12/' + 'tvn1749.pt'
    sqn = PATH + '/model/2023-09-12/' + 'sqn1749.pt'
    pn = PATH + '/model/2023-09-12/' + 'pn1749.pt'
    vnmodel = ValueNet(8, 256)
    vnmodel.load_state_dict(torch.load(vn))
    vnmodel.eval()
    tvnmodel = ValueNet(8, 256)
    tvnmodel.load_state_dict(torch.load(tvn))
    tvnmodel.eval()
    sqnmodel = SoftQNet(8, 2, 256)
    sqnmodel.load_state_dict(torch.load(sqn))
    sqnmodel.eval()
    pnmodel = PolicyNet(8, 2, 256)
    pnmodel.load_state_dict(torch.load(pn))
    pnmodel.eval()
    return vnmodel, tvnmodel, sqnmodel, pnmodel

class Config:
    def __init__(self):
        self.algo_name = 'SAC'
        self.reward_scale = 10 # 奖励尺度
        self.train_eps = 3000 # 训练迭代次数
        self.test_eps = 300 # 测试迭代次数
        self.max_steps = 200 # 每次迭代最大时间步
        self.max_insert_steps = 200
        self.gamma = 0.99 #折扣因子
        self.mean_lambda=1e-3 # 重参数化分布均值的损失权重
        self.std_lambda=1e-3 # 重参数化分布标准差的损失权重
        self.z_lambda=0.0 # 重参数化分布抽样值的损失权重
        self.soft_tau=1e-2 # 目标网络软更新系数
        self.value_lr  = 3e-4 # 值网络的学习率
        self.soft_q_lr = 3e-4 # Q网络的学习率
        self.policy_lr = 3e-4 # 策略网络的学习率
        self.insert_value_lr  = 3e-5 # 值网络的学习率
        self.insert_soft_q_lr = 3e-5 # Q网络的学习率
        self.insert_policy_lr = 3e-5 # 策略网络的学习率
        self.capacity = 1000000 # 经验缓存池的大小
        self.hidden_dim = 256 # 隐藏层维度
        self.batch_size  = 256 # 批次大小
        self.insert_batch_size = 256
        self.device=torch.device("cpu") # 使用设备

# tool0位置
# target = Pose()
# target.position.x = -0.524959331481
# target.position.y = -0.0986855744651
# target.position.z = 0.310585374262
# target.orientation.x = 0.711484940497
# target.orientation.y = 0.702694320621
# target.orientation.z = 0.0020990708527
# target.orientation.w = 0.0023377583519
target = Pose()
target.position.x = -0.529350149756
target.position.y = 0.0804129344187
target.position.z = 0.310885374262
target.orientation.x = 0.711484940497
target.orientation.y = 0.702694320621
target.orientation.z = 0.0020990708527
target.orientation.w = 0.0023377583519

cfg = Config()
env, agent, insert_agent = env_agent_config(cfg)
# res_dic = train(cfg, env, agent, target)
# res_dic = train_insert(cfg, env, agent, insert_agent, target)
test_dic = test(cfg, env, agent, insert_agent, target)
env.finish()
print("kannryou")