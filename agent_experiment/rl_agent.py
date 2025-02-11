import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import numpy as np

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(12, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, action_size)
        )
        
    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

class RLAgent:
    def __init__(self, state_size, action_size, personality_traits):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.personality = personality_traits
        
        self.policy_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.gamma = 0.99
        self.epsilon = 0.1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.tau = 0.001
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        self.stress_level = 0
        self.adaptation_score = 50
        self.social_connections = set()
        
        self.metrics = {
            'stress_history': [],
            'adaptation_history': [],
            'reward_history': [],
            'action_history': [],
            'q_values': []
        }
        
    def get_state_tensor(self, state_dict):
        """Convert state dictionary to tensor with enhanced social features"""
        confined = float(state_dict['environment'].get('confined', False))
        has_exit = float(state_dict['environment'].get('has_exit', False))
        crisis = float(state_dict['environment'].get('crisis_event', False))
        
        social_influence = state_dict.get('social_connections', 0) / 10.0
        social_stress = sum(c.stress_level for c in state_dict.get('others', [])) / 100.0
        group_adaptation = sum(c.adaptation_score for c in state_dict.get('others', [])) / 100.0
        
        state = [
            float(state_dict['stress_level']),
            float(state_dict['social_connections']),
            float(state_dict['adaptation_score']),
            confined,
            crisis,
            has_exit,
            social_influence,
            social_stress,
            group_adaptation,
            float(self.personality['extroversion']),
            float(self.personality['neuroticism']),
            float(self.personality['resilience'])
        ]
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def select_action(self, state_dict):
        """Select action using epsilon-greedy policy with personality influence"""
        state = self.get_state_tensor(state_dict)
        
        personality_factor = (
            self.personality['openness'] * 0.4 +
            -self.personality['neuroticism'] * 0.3 +
            self.personality['extroversion'] * 0.3
        )
        adjusted_epsilon = max(
            self.epsilon_min,
            self.epsilon * (1 + personality_factor)
        )
        
        if random.random() > adjusted_epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
                self.metrics['q_values'].append(q_values.cpu().numpy())
                return q_values.max(1)[1].item()
        return random.randrange(self.action_size)
    
    def update(self, state, action, reward, next_state, done):
        """Update agent's state and learning"""
        self.memory.append(Experience(
            self.get_state_tensor(state),
            torch.tensor([action]).to(self.device),
            torch.tensor([reward]).to(self.device),
            self.get_state_tensor(next_state),
            torch.tensor([done]).to(self.device)
        ))
        
        self.metrics['stress_history'].append(state['stress_level'])
        self.metrics['adaptation_history'].append(state['adaptation_score'])
        self.metrics['reward_history'].append(reward)
        self.metrics['action_history'].append(action)
        
        if len(self.memory) >= self.batch_size:
            self._learn()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _learn(self):
        """Learn from stored experiences"""
        experiences = random.sample(self.memory, self.batch_size)
        batch = Experience(*zip(*experiences))
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)
        
        current_q = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0]
            target_q = reward_batch + (1 - done_batch) * self.gamma * next_q
        
        loss = F.smooth_l1_loss(current_q, target_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )
    
    def get_metrics(self):
        """Return current metrics for analysis"""
        return {
            'stress_mean': np.mean(self.metrics['stress_history'][-100:]),
            'adaptation_mean': np.mean(self.metrics['adaptation_history'][-100:]),
            'reward_mean': np.mean(self.metrics['reward_history'][-100:]),
            'q_values_mean': np.mean(self.metrics['q_values'][-100:]),
            'action_distribution': np.bincount(
                self.metrics['action_history'][-100:],
                minlength=self.action_size
            ) / 100
        }
