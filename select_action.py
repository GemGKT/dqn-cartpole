import torch
import random
import numpy as np

def select_action(state, policy_net, epsilon, env):
    if random.random() < epsilon:
        return env.action_space.sample()  # Chọn random (exploration)
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)  # Thêm batch dimension
            q_values = policy_net(state)
            return q_values.argmax().item()  # Chọn action có Q-value lớn nhất