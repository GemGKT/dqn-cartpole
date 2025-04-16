import gym
import torch
import numpy as np
from model import DQN
import os
import imageio

# Tạo thư mục videos nếu chưa có
os.makedirs(r"E:\Codes\Reinforcement_Learning\Week_1\cartpole_dqn\videos", exist_ok=True)

# Khởi tạo môi trường ghi frame
env = gym.make("CartPole-v1", render_mode="rgb_array")
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# Load mô hình đã học
model = DQN(input_dim, output_dim)
model.load_state_dict(torch.load(r"E:\Codes\Reinforcement_Learning\Week_1\cartpole_dqn\best_dqn_cartpole.pt"))
model.eval()

# Reset môi trường
state, _ = env.reset()
done = False
total_reward = 0
frames = []

while not done:
    # Lấy frame hiện tại để ghi video
    frame = env.render()
    frames.append(frame)

    # Chọn hành động theo policy
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = model(state_tensor)
        action = q_values.argmax().item()

    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = next_state
    total_reward += reward

env.close()

# Ghi video
video_path = r"E:\Codes\Reinforcement_Learning\Week_1\cartpole_dqn\videos\demo_ep1.mp4"
imageio.mimsave(video_path, frames, fps=30)

# Thông tin kết quả
print("=" * 40)
print("🎮 DQN CartPole Demo")
print(f"🏆 Total Reward: {total_reward}")
print(f"🎥 Video saved to: {video_path}")
print("=" * 40)
