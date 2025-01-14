import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import sys, os
import seaborn as sns 
sys.path.append(os.path.abspath(os.curdir))  ## only for vscode debug
from rl_only_net import SAP, SAP_loss, ReplayBuffer
from PIL import Image, ImageDraw
import numpy as np
from env_test import UREnv
from torchviz import make_dot

TRAIN_MODE = True
VAL_FILE_IDX = 80
force_scale = 10.0

class Config:
    def __init__(self):
        self.algo_name = 'SAC'
        self.reward_scale = 10 # 奖励尺度
        self.train_eps = 1000 # 训练迭代次数
        self.test_eps = 100 # 测试迭代次数
        self.max_steps = 100 # 每次迭代最大时间步
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
        self.hidden_dim = 32 # 隐藏层维度
        self.device=torch.device("cpu") # 使用设备

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

def draw_spatial_features(numpy_image, features, image_size=(28, 28)):
    image_size_x, image_size_y = image_size
    img = Image.fromarray((numpy_image*255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    for sp in features:
        x, y = sp
        attend_x_pix = int((x + 1) * (image_size_x - 1) / 2)
        attend_y_pix = int((y + 1) * (image_size_y - 1) / 2)
               
        attend_y_pix = max(0, attend_y_pix)
        attend_x_pix = max(0, attend_x_pix)
        attend_y_pix = min(attend_y_pix, image_size_y-1)
        attend_x_pix = min(attend_x_pix, image_size_x-1)
        
        # numpy_image[attend_y_pix, attend_x_pix] = np.array([0.0, 0.0, 1.0])
        draw.ellipse((attend_x_pix-1.5, attend_y_pix-1.5, attend_x_pix+1.5, attend_y_pix+1.5), fill=(255,255,0))
    return (np.array(img)/255).astype(np.float32)


def draw_figure(filename, num_images_to_draw, spatial_features_to_draw, images_to_draw, reconstructed_images_to_draw):
    f, axarr = plt.subplots(num_images_to_draw, 2, figsize=(10, 15), dpi=100)
    plt.tight_layout()
    for idx, im in enumerate(reconstructed_images_to_draw[:num_images_to_draw]):
        # original image
        og_image = (images_to_draw[:num_images_to_draw][idx] + 1) / 2
        og_im_res = np.repeat(og_image.numpy().reshape(64, 64, 1), 3, axis=2)
        img =  draw_spatial_features(og_im_res, spatial_features_to_draw[idx], image_size=(64, 64))
        # axarr[idx, 0].imshow(og_im_res)
        axarr[idx, 0].imshow(img)
        # reconstructed image
        scaled_image = (im + 1) / 2
        axarr[idx, 1].imshow(scaled_image.detach().numpy().reshape(64, 64), cmap="gray")

    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--file_name", type=str, default=os.path.join(os.path.abspath("robot_driver/results"), "out_covT_heatmap3.png"))
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--weight_path", type=str, default="src/robot_driver/weight")
    args = parser.parse_args()
    
    # parameters and miscellaneous
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    # Adam learning rate
    lr = args.learning_rate
    out_file_name = args.file_name
    mode = args.mode
    weight_path = args.weight_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = UREnv(0)
    cfg = Config()
    n_states = env.observation_space.shape[0]
    n_inserts = env.insert_space.shape[0]
    n_actions = env.action_space.shape[0]
    print(f"状态空间维度：{n_states}，动作空间维度：{n_actions}")
    # 更新n_states和n_actions到cfg参数中
    setattr(cfg, 'n_states', n_states)
    setattr(cfg, 'n_actions', n_actions) 
    setattr(cfg, 'n_inserts', n_inserts) 
    setattr(cfg, 'batch_size', batch_size) 
    setattr(cfg, 'action_space', env.action_space) 

    memory = ReplayBuffer(cfg.capacity)
    sap_model = SAP(cfg, in_channels=1, encoder_out_channels=(4, 8, 8), decoder_input_size=(58,58)).to(device)
    optimiser = torch.optim.Adam(sap_model.parameters(), lr=lr)
    sap_loss = SAP_loss(0.1)
    
    rewards_in_round = []
    ma_rewards_in_round = []
    if TRAIN_MODE:
        for epoch in range(num_epochs):
            sap_model.train()
            state = env.reset()
            reward_in_round = 0
            loss = gi = ga = 0
            for step in range(cfg.max_steps):
                action = sap_model(state)
                # np.set_printoptions(precision=4, suppress=True)
                # print(state_mem)
                
                next_state, reward, done = env.step(force_scale*action)
                reward = reward*cfg.reward_scale
                memory.push(state, action, reward, next_state, done)
                state = next_state
                reward_in_round += reward

                if done or step==cfg.max_steps-1:
                # 更新网络
                    rewards_in_round.append(reward_in_round)
                    if ma_rewards_in_round:
                        ma_rewards_in_round.append(0.9*ma_rewards_in_round[-1]+0.1*reward_in_round) 
                    else:
                        ma_rewards_in_round.append(reward_in_round)
                    plot_rewards(rewards_in_round, ma_rewards_in_round)
                    if len(memory) >= cfg.batch_size:
                        for i in range(step):
                            state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(cfg.batch_size)
                            state_batch      = torch.FloatTensor(state_batch).to(cfg.device)
                            next_state_batch = torch.FloatTensor(next_state_batch).to(cfg.device)
                            action_batch     = torch.FloatTensor(action_batch).to(cfg.device)
                            reward_batch     = torch.FloatTensor(reward_batch).unsqueeze(1).to(cfg.device)
                            done_batch       = torch.FloatTensor(np.float32(done_batch)).unsqueeze(1).to(cfg.device)
                            
                            loss_rl = sap_model.sac.update(cfg.gamma, cfg.mean_lambda, cfg.std_lambda, cfg.z_lambda, 
                                                state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                            print('Train Epoch: {} Loss: {:.6f}, r: {}'.format(epoch, loss_rl, reward_in_round))
                    break
            # if epoch % 10 == 0 and epoch != 0:
            #     torch.save(sap_model.state_dict(),weight_path + "/weight_" + str(epoch) + ".plk")
            #     print('保存成功')
    num_images = 10
    state_batch, _, _, _, _, image_batch, _ = memory.sample(cfg.batch_size)
    image_batch = torch.FloatTensor(image_batch).to(cfg.device)
    state_batch = torch.FloatTensor(state_batch).to(cfg.device)
    key_points, image_predict = sap_model.draw_fig(state_batch, image_batch)
    key_points = torch.FloatTensor(key_points).to(cfg.device)
    draw_figure(out_file_name, num_images, key_points.to("cpu"), image_batch.to("cpu"), image_predict.to("cpu"))
    print("f")
