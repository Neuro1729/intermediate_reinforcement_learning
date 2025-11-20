import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.mpe import simple_spread_v2

# --- Actor & Critic definitions ---
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Linear(obs_dim, 128)
        self.out = nn.Linear(128, act_dim)
    def forward(self, obs):
        x = torch.relu(self.fc(obs))
        return torch.tanh(self.out(x))  # continuous

class Critic(nn.Module):
    def __init__(self, joint_obs_dim, joint_act_dim):
        super().__init__()
        self.fc = nn.Linear(joint_obs_dim + joint_act_dim, 128)
        self.out = nn.Linear(128, 1)
    def forward(self, obs, acts):
        x = torch.relu(self.fc(torch.cat([obs, acts], dim=-1)))
        return self.out(x)

# --- MADDPG Setup ---
env = simple_spread_v2.parallel_env(N=3, local_ratio=0.5)  # say 3 agents
env.reset()

num_agents = len(env.agents)
obs_dim = env.observation_spaces[env.agents[0]].shape[0]
act_dim = env.action_spaces[env.agents[0]].shape[0]

actors = {agent: Actor(obs_dim, act_dim) for agent in env.agents}
critics = {agent: Critic(obs_dim * num_agents, act_dim * num_agents) for agent in env.agents}

actor_opts = {agent: optim.Adam(actors[agent].parameters(), lr=1e-3) for agent in env.agents}
critic_opts = {agent: optim.Adam(critics[agent].parameters(), lr=1e-3) for agent in env.agents}

gamma = 0.95

# --- Training Loop ---
for episode in range(200):
    obs = env.reset()  # dict: {agent: obs}
    done = {agent: False for agent in env.agents}

    while not all(done.values()):
        # 1. Actors choose actions
        actions = {}
        for agent in env.agents:
            obs_tensor = torch.tensor(obs[agent], dtype=torch.float32)
            actions[agent] = actors[agent](obs_tensor).detach().numpy()

        # 2. Step environment
        next_obs, rewards, dones, _ = env.step(actions)

        # 3. Critic updates
        # Build joint observation and action tensors
        joint_obs = torch.cat([torch.tensor(obs[a], dtype=torch.float32) for a in env.agents])
        joint_acts = torch.cat([torch.tensor(actions[a], dtype=torch.float32) for a in env.agents])
        joint_next_obs = torch.cat([torch.tensor(next_obs[a], dtype=torch.float32) for a in env.agents])

        for agent in env.agents:
            # compute target Q
            with torch.no_grad():
                next_actions = []
                for a in env.agents:
                    next_o = torch.tensor(next_obs[a], dtype=torch.float32)
                    next_actions.append(actors[a](next_o))
                joint_next_acts = torch.cat(next_actions)

                target_q = rewards[agent] + gamma * critics[agent](joint_next_obs, joint_next_acts)

            current_q = critics[agent](joint_obs, joint_acts)
            critic_loss = nn.MSELoss()(current_q, target_q.unsqueeze(-1))

            critic_opts[agent].zero_grad()
            critic_loss.backward()
            critic_opts[agent].step()

        # 4. Actor updates
        for agent in env.agents:
            # compute actor loss via its critic
            obs_tensor = torch.tensor(obs[agent], dtype=torch.float32)
            action_pred = actors[agent](obs_tensor)

            # replace this agent's action in joint actions
            joint_act_pred = []
            for a in env.agents:
                if a == agent:
                    joint_act_pred.append(action_pred)
                else:
                    # use the old action (no grad)
                    joint_act_pred.append(torch.tensor(actions[a], dtype=torch.float32))
            joint_act_pred = torch.cat(joint_act_pred)

            actor_loss = -critics[agent](joint_obs, joint_act_pred)

            actor_opts[agent].zero_grad()
            actor_loss.backward()
            actor_opts[agent].step()

        obs = next_obs
        done = dones

    print(f"Episode {episode} done")
