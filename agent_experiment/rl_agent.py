import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import numpy as np
from typing import Dict, List, Tuple
import torch.optim as optim

Experience = namedtuple(
    'Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class RLAgent:
    def __init__(self, state_size: int, action_size: int, personality_traits: Dict):
        self.state_size = state_size
        self.action_size = action_size
        self.personality = personality_traits
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.memory = deque(maxlen=2000)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        self.metrics = {
            'stress_mean': 0,
            'adaptation_mean': 0,
            'reward_mean': 0,
            'loss_history': []
        }
    
    def select_action(self, state: Dict) -> int:
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
            
        with torch.no_grad():
            state_tensor = self._process_state(state)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def _process_state(self, state: Dict) -> torch.Tensor:
        """Convert state dict to tensor"""
        state_list = [
            state['stress_level'],
            state['social_connections'],
            state['adaptation_score'],
            float(state['environment'].get('confined', False)),
            float(state['environment'].get('has_exit', False)),
            float(state['environment'].get('crisis_event', False)),
            self.personality['extroversion'],
            self.personality['resilience']
        ]
        return torch.FloatTensor(state_list).unsqueeze(0).to(self.device)
    
    def remember(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        
        self.metrics['stress_mean'] = 0.95 * self.metrics['stress_mean'] + 0.05 * state['stress_level']
        self.metrics['adaptation_mean'] = 0.95 * self.metrics['adaptation_mean'] + 0.05 * state['adaptation_score']
        self.metrics['reward_mean'] = 0.95 * self.metrics['reward_mean'] + 0.05 * reward
    
    def learn(self, state: Dict, reward: float):
        """Update policy based on state and reward"""
        self.remember(state, self.select_action(state), reward, state, False)
        self._learn_from_memory()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _learn_from_memory(self):
        """Train on batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.cat([self._process_state(s) for s, _, _, _, _ in batch])
        actions = torch.tensor([a for _, a, _, _, _ in batch], device=self.device)
        rewards = torch.tensor([r for _, _, r, _, _ in batch], device=self.device)
        next_states = torch.cat([self._process_state(s) for _, _, _, s, _ in batch])
        dones = torch.tensor([d for _, _, _, _, d in batch], device=self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones.float()))
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.metrics['loss_history'].append(loss.item())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if len(self.memory) % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def get_metrics(self) -> Dict:
        """Return current performance metrics"""
        return self.metrics

    def update(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        """Convenience method to maintain backward compatibility"""
        self.remember(state, action, reward, next_state, done)
        self._learn_from_memory()
