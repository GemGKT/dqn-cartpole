import random
from collections import deque, namedtuple

# Cấu trúc lưu 1 transition
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        """Lưu 1 transition vào buffer"""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        """Trả về 1 minibatch random"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
