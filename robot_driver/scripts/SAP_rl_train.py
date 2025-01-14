import argparse

from matplotlib import pyplot as plt
import numpy as np
import torch
import sys, os
import getpass
import seaborn as sns 
sys.path.append(os.path.abspath(os.curdir))  ## only for vscode debug
from SAP_rl_net import SAP, SAP_loss, ReplayTree, ReplayBuffer
from PIL import Image, ImageDraw
import numpy as np
from env_test import UREnv
from torchviz import make_dot

TRAIN_MODE = True
per_replay = False
VAL_FILE_IDX = 80
force_scale = 10.0
USER = getpass.getuser()

class Config:
    def __init__(self):
        self.algo_name = 'SAC'
        self.reward_scale = 10 # 奖励尺度
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
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--file_name", type=str, default=os.path.join(os.path.abspath("robot_driver/results"), "out_covT_heatmap3.png"))
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--weight_path", type=str, default="/home/"+USER+"/att_ws/src/attention2/robot_driver/weight")
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

    env = UREnv(8*2)
    cfg = Config()
    n_states = env.observation_space.shape[0]
    n_inserts = env.insert_space.shape[0]
    n_policy = env.policy_space.shape[0]
    n_actions = env.action_space.shape[0]
    print(f"状态空间维度：{n_states}，动作空间维度：{n_actions}")
    # 更新n_states和n_actions到cfg参数中
    setattr(cfg, 'n_states', n_states)
    setattr(cfg, 'n_actions', n_actions) 
    setattr(cfg, 'n_inserts', n_inserts) 
    setattr(cfg, 'batch_size', batch_size) 
    setattr(cfg, 'n_policy', n_policy) 

    if per_replay:
        memory = ReplayTree(cfg.capacity)
    else:
        memory = ReplayBuffer(cfg.capacity)
    sap_model = SAP(cfg, in_channels=1, encoder_out_channels=(4, 8, 8), decoder_input_size=(22,22)).to(device)
    optimiser = torch.optim.Adam(sap_model.parameters(), lr=lr)
    sap_loss = SAP_loss(num_epochs)
    
    rewards_in_round = []
    ma_rewards_in_round = []
    if TRAIN_MODE:
        for epoch in range(num_epochs):
            sap_model.train()
            state = env.reset()
            image = env.get_image()
            reward_in_round = 0
            loss = gi = ga = 0
            for step in range(cfg.max_steps):
                action, state_per = sap_model(state, image)
                next_state, reward, done = env.step(force_scale*action)
                next_image = env.get_image()
                reward = reward*cfg.reward_scale

                if per_replay:
                    policy_val = sap_model.sac.value_net(torch.FloatTensor(state_per).unsqueeze(0))
                    target_val = sap_model.sac.target_value_net(torch.FloatTensor(state_per))
                    if done:
                        error = abs(policy_val - reward)
                    else:
                        error = abs(policy_val - reward - cfg.gamma * target_val)
                    memory.push(error.cpu().detach().numpy(), (state, action, reward, next_state, done, 
                                                            image.squeeze(1).detach().numpy(), next_image.squeeze(1).detach().numpy()))
                else:
                    memory.push(state, action, reward, next_state, done, image.squeeze(1).detach().numpy(), next_image.squeeze(1).detach().numpy())
                
                state = next_state
                image = next_image
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
                            if per_replay:
                                (state_batch, action_batch, reward_batch, next_state_batch, done_batch, 
                                        image_batch, next_image_batch), idxs_batch, is_weights_batch = memory.sample(cfg.batch_size)
                            else:
                                (state_batch, action_batch, reward_batch, next_state_batch, done_batch, 
                                        image_batch, next_image_batch) = memory.sample(cfg.batch_size)
                            state_batch      = torch.FloatTensor(state_batch).to(cfg.device)
                            next_state_batch = torch.FloatTensor(next_state_batch).to(cfg.device)
                            action_batch     = torch.FloatTensor(action_batch).to(cfg.device)
                            reward_batch     = torch.FloatTensor(reward_batch).unsqueeze(1).to(cfg.device)
                            done_batch       = torch.FloatTensor(np.float32(done_batch)).unsqueeze(1).to(cfg.device)
                            image_batch      = torch.FloatTensor(image_batch).to(cfg.device)
                            next_image_batch = torch.FloatTensor(next_image_batch).to(cfg.device)
                            attention_points, attention_points_pre, state_batch = sap_model.get_attention(state_batch, image_batch)
                            attention_points_pre = torch.FloatTensor(attention_points_pre).to(cfg.device)
                            
                            image_pre = sap_model.get_image_prediction(attention_points_pre, image_batch)
                            loss_sap, gi, gp = sap_loss(image_pre, next_image_batch, attention_points_pre, attention_points, epoch)

                            state_batch = torch.FloatTensor(state_batch).to(cfg.device)
                            next_state_batch = torch.FloatTensor(sap_model.get_next_state(next_state_batch, next_image_batch)).to(cfg.device)
                            loss_rl, expected_value, next_value = sap_model.sac.update(cfg.gamma, cfg.mean_lambda, cfg.std_lambda, cfg.z_lambda, 
                                                                                           state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                            loss = loss_rl + loss_sap

                            if per_replay:
                                abs_errors = np.sum(np.abs(expected_value.cpu().detach().numpy() - next_value.cpu().detach().numpy()), axis=1)
                                memory.batch_update(idxs_batch, abs_errors) 

                            optimiser.zero_grad()
                            sap_model.sac.policy_optimizer.zero_grad()
                            sap_model.sac.value_optimizer.zero_grad()
                            sap_model.sac.soft_q_optimizer.zero_grad()
                            # dot = make_dot(loss.mean())
                            # dot.render("model.pdf")
                            loss.backward()
                            optimiser.step()
                            sap_model.sac.policy_optimizer.step()
                            sap_model.sac.value_optimizer.step()
                            sap_model.sac.soft_q_optimizer.step()
                            sap_model.sac.update_target_v()

                            print('Train Epoch: {} Loss: {:.6f}, rl: {:.6f}, gi: {:.6f}, gp: {:.6f}, r: {}'.format(epoch, loss, loss_rl, gi, gp, reward_in_round))

                        if epoch == 0 or (epoch+1)%100 == 0:
                            num_images = 10
                            if per_replay:
                                (state_batch, action_batch, reward_batch, next_state_batch, done_batch, image_batch, next_image_batch), 
                                idxs_batch, is_weights_batch = memory.sample(cfg.batch_size)
                            else:
                                (state_batch, action_batch, reward_batch, next_state_batch, done_batch, image_batch, next_image_batch) = memory.sample(cfg.batch_size)
                            image_batch = torch.FloatTensor(image_batch).to(cfg.device)
                            state_batch = torch.FloatTensor(state_batch).to(cfg.device)
                            key_points, image_predict = sap_model.draw_fig(state_batch, image_batch)
                            key_points = torch.FloatTensor(key_points).to(cfg.device)
                            draw_figure(out_file_name, num_images, key_points.to("cpu"), image_batch.to("cpu"), image_predict.to("cpu"))
                            torch.save(sap_model.state_dict(), weight_path + "/weight_" + str(epoch) + ".plk")
                            sap_model.sac.save_model(epoch)
                            memory.save_buffer(epoch)
                    
                    break

    # num_images = 10
    # if per_replay:
    #     (state_batch, action_batch, reward_batch, next_state_batch, done_batch, image_batch, next_image_batch), 
    #     idxs_batch, is_weights_batch = memory.sample(cfg.batch_size)
    # else:
    #     (state_batch, action_batch, reward_batch, next_state_batch, done_batch, image_batch, next_image_batch) = memory.sample(cfg.batch_size)
    # image_batch = torch.FloatTensor(image_batch).to(cfg.device)
    # state_batch = torch.FloatTensor(state_batch).to(cfg.device)
    # key_points, image_predict = sap_model.draw_fig(state_batch, image_batch)
    # key_points = torch.FloatTensor(key_points).to(cfg.device)
    # draw_figure(out_file_name, num_images, key_points.to("cpu"), image_batch.to("cpu"), image_predict.to("cpu"))
    print("f")
