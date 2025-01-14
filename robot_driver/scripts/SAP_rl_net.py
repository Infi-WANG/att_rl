import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from torch import nn
from torch.distributions import Normal
import time
import pickle
import getpass

USER = getpass.getuser()

class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data_pointer = 0
        self.n_entries = 0
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype = object)

    def update(self, tree_idx, p):
        '''Update the sampling weight
        '''
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, p, data):
        '''Adding new data to the sumTree
        '''
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        # print ("tree_idx=", tree_idx)
        # print ("nonzero = ", np.count_nonzero(self.tree))
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_leaf(self, v):
        '''Sampling the data
        '''
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx] :
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total(self):
        return int(self.tree[0])
    
class ReplayTree:
    '''ReplayTree for the per(Prioritized Experience Replay) DQN. 
    '''
    def __init__(self, capacity):
        self.capacity = capacity # the capacity for memory replay
        self.tree = SumTree(capacity)
        ## hyper parameter for calculating the importance sampling weight
        self.beta_increment_per_sampling = 0.001
        self.alpha = 0.6
        self.beta = 0.4
        self.epsilon = 0.01 
        self.abs_err_upper = 1.

    def __len__(self):
        ''' return the num of storage
        '''
        return self.tree.total()

    def push(self, error, sample):
        '''Push the sample into the replay according to the importance sampling weight
        '''
        p = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.add(p, sample)         

    def sample(self, batch_size):
        '''This is for sampling a batch data and the original code is from:
        https://github.com/rlcode/per/blob/master/prioritized_memory.py
        '''
        pri_segment = self.tree.total() / batch_size

        priorities = []
        batch = []
        idxs = []

        is_weights = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total() 

        for i in range(batch_size):
            a = pri_segment * i
            b = pri_segment * (i+1)

            s = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
            prob = p / self.tree.total()

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return zip(*batch), idxs, is_weights
    
    def batch_update(self, tree_idx, abs_errors):
        '''Update the importance sampling weight
        '''
        abs_errors += self.epsilon

        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class CoordinateUtils(object):
    @staticmethod
    def get_image_coordinates(h, w, normalise):
        x_range = torch.arange(w, dtype=torch.float32)
        y_range = torch.arange(h, dtype=torch.float32)
        if normalise:
            x_range = (x_range / (w - 1)) * 2 - 1
            y_range = (y_range / (h - 1)) * 2 - 1
        image_x = x_range.unsqueeze(0).repeat_interleave(h, 0)
        image_y = y_range.unsqueeze(0).repeat_interleave(w, 0).t()
        return image_x, image_y

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

        self.mean_p = nn.Linear(hidden_dim, 16)
        self.mean_p.weight.data.uniform_(-init_w, init_w)
        self.mean_p.bias.data.uniform_(-init_w, init_w)
        self.log_std_p = nn.Linear(hidden_dim, 16)
        self.log_std_p.weight.data.uniform_(-init_w, init_w)
        self.log_std_p.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        mean_p = self.mean_p(x)
        log_std_p = self.log_std_p(x)
        log_std_p = torch.clamp(log_std_p, self.log_std_min, self.log_std_max)

        return mean, log_std, mean_p, log_std_p
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std, _, _ = self.forward(state)
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
        mean, log_std, _, _ = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        action = action.detach().cpu().numpy()
        return action[0]

    def get_action_p(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        _, _, mean_p, log_std_p = self.forward(state)
        std = log_std_p.exp()

        normal = Normal(mean_p, std)
        z = normal.sample()
        action = torch.tanh(z)

        action = action.detach().cpu().numpy()
        return action[0]

    def evaluate_p(self, state, epsilon=1e-6):
        _, _, mean_p, log_std_p = self.forward(state)
        std = log_std_p.exp()
        ## 计算动作
        normal = Normal(mean_p, std)
        z = normal.sample()
        action = torch.tanh(z)
        ## 计算动作概率
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob, z, mean_p, log_std_p
    
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

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.positon = 0

    def push(self, state, action, reward, next_state, done, image, next_image):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.positon] = (state, action, reward, next_state, done, image, next_image)
        self.positon = (self.positon + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # 随机采出小批量转移
        state, action, reward, next_state, done, image, next_image =  zip(*batch) # 解压成状态，动作等
        return state, action, reward, next_state, done, image, next_image
    
    def save_buffer(self, epoch):
        with open(time.strftime("%Y-%m-%d %H:%M", time.localtime()) +' replay_buffer' + str(epoch) + '.pkl', 'wb') as f:
            pickle.dump(self.buffer, f)
    
    def __len__(self):
        return len(self.buffer)
    
class SAC:
    def __init__(self, cfg) -> None:
        self.n_states = cfg.n_states
        self.n_actions = cfg.n_actions
        self.n_policy = cfg.n_policy
        self.device = cfg.device
        self.value_net = ValueNet(self.n_states, cfg.hidden_dim).to(self.device)
        self.target_value_net = ValueNet(self.n_states, cfg.hidden_dim).to(self.device)
        self.soft_q_net = SoftQNet(self.n_states, self.n_actions, cfg.hidden_dim).to(self.device)
        self.policy_net = PolicyNet(self.n_states, self.n_policy, cfg.hidden_dim).to(self.device)
        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=cfg.value_lr)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=cfg.soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.policy_lr)
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)
        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()

    def update(self, gamma, mean_lambda, std_lambda, z_lambda,
               state, action, reward, next_state, done):
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

        loss_rl = q_value_loss + value_loss + policy_loss
        # print('q: {:.6f}, v: {:.6f}, p: {:.6f}'.format(q_value_loss, value_loss, policy_loss))
        return loss_rl, expected_value, next_q_value
    
    def update_target_v(self, soft_tau=1e-2):
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    def save_model(self, epoch):
        PATH = "/home/"+USER+"/att_ws/src/attention2/robot_driver/weight"
        vPATH = PATH + '/' + time.strftime("%Y-%m-%d %H:%M", time.localtime()) +' vn' + str(epoch) + '.pt'
        tvPATH = PATH + '/' + time.strftime("%Y-%m-%d %H:%M", time.localtime()) +' tvn' + str(epoch) + '.pt'
        sqPATH = PATH + '/' + time.strftime("%Y-%m-%d %H:%M", time.localtime()) +' sqn' + str(epoch) + '.pt'
        pPATH = PATH + '/' + time.strftime("%Y-%m-%d %H:%M", time.localtime()) +' pn' + str(epoch) + '.pt'
        torch.save(self.value_net.state_dict(), vPATH)
        torch.save(self.target_value_net.state_dict(), tvPATH)
        torch.save(self.soft_q_net.state_dict(), sqPATH)
        torch.save(self.policy_net.state_dict(), pPATH)

class SpatialSoftArgmax(nn.Module):
    def __init__(self, temperature=None, normalise=False):
        """
        Applies a spatial soft argmax over the input images.
        :param temperature: The temperature parameter (float). If None, it is learnt.
        :param normalise: Should spatial features be normalised to range [-1, 1]?
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1)) if temperature is None else torch.tensor([temperature])
        self.normalise = normalise

    def forward(self, x):
        """
        Applies Spatial SoftArgmax operation on the input batch of images x.
        :param x: batch of images, of size (N, C, H, W)
        :return: Spatial features (one point per channel), of size (N, C, 2)
        """
        n, c, h, w = x.size()
        spatial_softmax_per_map = nn.functional.softmax(x.view(n * c, h * w) / self.temperature, dim=1)
        spatial_softmax = spatial_softmax_per_map.view(n, c, h, w)

        # calculate image coordinate maps
        image_x, image_y = CoordinateUtils.get_image_coordinates(h, w, normalise=self.normalise)
        # size (H, W, 2)
        image_coordinates = torch.cat((image_x.unsqueeze(-1), image_y.unsqueeze(-1)), dim=-1)
        # send to device
        image_coordinates = image_coordinates.to(device=x.device)

        # multiply coordinates by the softmax and sum over height and width, like in [2]
        expanded_spatial_softmax = spatial_softmax.unsqueeze(-1)  ## (256, 8, 3, 3, 1)
        image_coordinates = image_coordinates.unsqueeze(0)  ## (1, 3, 3, 2), (3,3)对应的是x，和y的坐标，乘以对应的softmax
        out = torch.sum(expanded_spatial_softmax * image_coordinates, dim=[2, 3])  ## 这里是元素相乘，目标是每个元素的softmax值分别乘以其x、y坐标 延展： (256, 8, 3, 3, 1) => (256, 8, 3, 3, 2), (1, 3, 3, 2) => (1, 1, 3, 3, 2)，在2,3维度求和
        # (N, C, 2)
        return out


class Heatmap(nn.Module):
    def __init__(self, img_height, img_width, sigma = 5) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_width = img_width
        self.img_height = img_height
        self.sigma = sigma
        
        X1 = torch.linspace(1, self.img_width, self.img_width)
        Y1 = torch.linspace(1, self.img_height, self.img_height)
        [self.X, self.Y] = torch.meshgrid(X1, Y1, indexing='xy')
        self.X = self.X.to(self.device)
        self.Y = self.Y.to(self.device)  
        
    def forward(self, x):
        # x: B, n_p, 2
        # out: B, n_p, h, w
        # out = torch.zeros(x.shape[0], x.shape[1], self.img_height, self.img_width).to(self.device)
        out_batch = []
        for batch_idx in range(x.shape[0]):
            out_ = []
            for point_idx in range(x.shape[1]):
                p_x = x[batch_idx, point_idx, 0]
                p_y = x[batch_idx, point_idx, 1] 
                attend_x_pix = (p_x + 1) * (self.img_height - 1) / 2
                attend_y_pix = (p_y + 1) * (self.img_width - 1) / 2
                X_ = self.X - attend_x_pix
                Y_ = self.Y - attend_y_pix
                D2 = X_ * X_ + Y_ * Y_
                E2 = 2.0 * self.sigma * self.sigma 
                Exponent = D2 / E2
                out_.append(torch.exp(-Exponent))
                
                # out[batch_idx, point_idx, :, :] = torch.exp(-Exponent)
            out_batch.append(torch.stack(out_))
        return torch.stack(out_batch)

class SAP(nn.Module):
    
    '''
        :param out_info_size: 6 for wrench only, 3 for position only, 9 for both 
        :param in_info_size:  9 for wrench only, 6 for position only, 12 for both (since actions take 3) 
    '''
    def __init__(self, cfg, in_channels, encoder_out_channels, decoder_input_size, temperature=None):
        super().__init__()
        self.image_feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=encoder_out_channels[0], kernel_size=5, stride=2),
            nn.BatchNorm2d(encoder_out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels=encoder_out_channels[0], out_channels=encoder_out_channels[1], kernel_size=5),
            nn.BatchNorm2d(encoder_out_channels[1]),
            nn.ReLU(),
            nn.Conv2d(in_channels=encoder_out_channels[1], out_channels=encoder_out_channels[2], kernel_size=5),
            nn.BatchNorm2d(encoder_out_channels[2]),
            nn.ReLU()
        ) ## -> (B, encoder_out_channels[2], out_H, out_W), (out_H, out_W)应与decoder_input_size相等
        
        self.feature_area_extractor_argmax = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=encoder_out_channels[0], kernel_size=5, stride=2),
            nn.BatchNorm2d(encoder_out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels=encoder_out_channels[0], out_channels=encoder_out_channels[1], kernel_size=5),
            nn.BatchNorm2d(encoder_out_channels[1]),
            nn.ReLU(),
            nn.Conv2d(in_channels=encoder_out_channels[1], out_channels=encoder_out_channels[2], kernel_size=5),
            nn.BatchNorm2d(encoder_out_channels[2]),
            nn.ReLU(),
            SpatialSoftArgmax(temperature=temperature, normalise=True)
        ) ## -> (B, encoder_out_channels[2], 2)
        
        self.sac = SAC(cfg)
        self.heatmap = Heatmap(decoder_input_size[0], decoder_input_size[1], sigma=10) ## -> (B, encoder_out_channels[2], out_H, out_W)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_out_channels[2], 8, kernel_size=5),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=6, stride=2),
            nn.Tanh()
        )

    def forward(self, state, image): # i: (32, 1, 200, 200), a: (32, 15)
        attention_points = self.feature_area_extractor_argmax(image)
        state = self.state_append_ap(state, attention_points)

        action = self.sac.policy_net.get_action(state)
        low  = -1
        high = 1
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action, state
    
    def get_attention(self, state, image):
        n = state.shape[0]
        attention_points = self.feature_area_extractor_argmax(image)
        ap = attention_points.reshape(n, -1).squeeze(0)
        state_batch = torch.cat((state, ap), dim=1)

        action, _, _, _, _ = self.sac.policy_net.evaluate_p(state_batch)
        attention_points_pre = action.reshape(n,-1,2)
        
        return attention_points, attention_points_pre, state_batch

    def get_image_prediction(self, ap_pre, image):
        return self.decoder(self.image_feature_extractor(image)*self.heatmap(ap_pre))
    
    def get_next_state(self, next_state, next_image):
        n = next_state.shape[0]
        attention_points = self.feature_area_extractor_argmax(next_image)
        ap = attention_points.reshape(n, -1).squeeze(0).detach().numpy()
        next_state = np.concatenate([next_state, ap], axis=1)

        return next_state
    
    def state_append_ap(self, state, ap):
        n = ap.shape[0]
        ap = ap.reshape(n, -1).squeeze(0).detach().numpy()
        state = np.append(state, ap)
        return state
    
    def draw_fig(self, state, image):
        state = self.get_next_state(state, image)
        action = self.sac.policy_net.get_action_p(state)

        n = action.shape[0]
        attention_points_pre = action.reshape(n,-1,2)
        image_pre = self.get_image_prediction(attention_points_pre, image)

        return attention_points_pre, image_pre
    
class SAP_loss(object):

    def __init__(self, epoch):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.weight_i_up = 100
        self.weight_p_up = 100
        self.weight_i_low = 10
        self.weight_p_low = 10
        self.epoch = epoch
        
    def __call__(self, reconstructed, target, att_points_plus1, att_points, epoch):
        """
        Performs the loss computation, and returns both loss components.
        :param reconstructed: Reconstructed, grayscale image
        :param target: Target, grayscale image
        :param a_hat: Predicted info
        :param a: target into
        :param ft: Features produced by the encoder for the target image
        :param ft_plus1: Features produced by the encoder for the next image in the trajectory to the target one
        :return: Loss
        """ 
        weight_i = self.weight_i_low + epoch*(self.weight_i_up - self.weight_i_low)/500
        weight_p = self.weight_p_up - epoch*(self.weight_p_up - self.weight_p_low)/500
        gi = self.mse_loss(reconstructed, target)/(len(reconstructed)*64*64)*weight_i
        gp = self.mse_loss(att_points_plus1, att_points)/(len(reconstructed)*8*2)*weight_p
        return gi + gp, gi, gp
