import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import DQN
from replay_buffer import ReplayBuffer
from select_action import select_action  # nếu bạn tách riêng, còn không thì paste vào đây luôn
import matplotlib.pyplot as plt

# Hyperparameters
env = gym.make("CartPole-v1")
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

episode_rewards = []
best_reward = 0
lr = 1e-3
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
buffer_size = 10000
batch_size = 128
target_update = 10
episodes = 500

# Mạng chính và mạng mục tiêu
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())  # Khởi tạo giống nhau
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
criterion = nn.MSELoss()
replay_buffer = ReplayBuffer(buffer_size)

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for t in range(200):
        action = select_action(state, policy_net, epsilon, env)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Lưu trải nghiệm
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Train nếu đủ dữ liệu
        if len(replay_buffer) > batch_size:
            transitions = replay_buffer.sample(batch_size)
            batch = list(zip(*transitions))

            states = torch.FloatTensor(batch[0])
            actions = torch.LongTensor(batch[1]).unsqueeze(1)
            rewards = torch.FloatTensor(batch[2]).unsqueeze(1)
            next_states = torch.FloatTensor(batch[3])
            dones = torch.FloatTensor(batch[4]).unsqueeze(1)

            # Tính Q hiện tại và Q mục tiêu
            q_values = policy_net(states).gather(1, actions)
            next_q_values = target_net(next_states).max(1)[0].detach().unsqueeze(1)
            expected_q = rewards + gamma * next_q_values * (1 - dones)

            loss = criterion(q_values, expected_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if done:
            break

    # Giảm epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Cập nhật target net mỗi target_update tập
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Total reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

     # Tổng kết sau episode
    episode_rewards.append(total_reward)
    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(policy_net.state_dict(), "best_dqn_cartpole.pt")
env.close()

# Vẽ biểu đồ
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN CartPole Reward Progress")
plt.grid()
plt.savefig("reward_plot.png")
plt.show()