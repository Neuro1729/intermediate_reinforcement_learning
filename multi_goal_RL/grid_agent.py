'''
Gridworld 5x5
Goals:
Coin1 (+1)
Coin2 (+1)
Key (+2)
Exit (+5, only if key collected)
State: [agent_x, agent_y, coin1_flag, coin2_flag, key_flag]
Reward: sum of collected goal rewards

Episode ends: max steps reached or exit collected

Challenge:
Agent must learn that collecting the key before exit is required
May also prefer picking nearby coins first to maximize reward
'''
import gym
from gym import spaces
import numpy as np
import random

# ---------------------------
# 1. Multi-Goal Planning Gridworld Environment
# ---------------------------
class MultiGoalGridEnv(gym.Env):
    def __init__(self, size=5, max_steps=20):
        super(MultiGoalGridEnv, self).__init__()
        self.size = size
        self.max_steps = max_steps

        # Actions: 0=up,1=down,2=left,3=right
        self.action_space = spaces.Discrete(4)

        # Observation: agent_x, agent_y, coin1_flag, coin2_flag, key_flag
        self.observation_space = spaces.MultiDiscrete([size, size, 2, 2, 2])

        self.reset()

    def reset(self):
        # Agent position
        self.agent_pos = [random.randint(0, self.size-1), random.randint(0, self.size-1)]

        # Goal positions
        self.coin1_pos = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
        self.coin2_pos = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
        self.key_pos = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
        self.exit_pos = [self.size-1, self.size-1]  # fixed exit

        # Avoid overlapping start positions
        occupied = [self.agent_pos]
        for pos in [self.coin1_pos, self.coin2_pos, self.key_pos]:
            while pos in occupied:
                pos[0] = random.randint(0, self.size-1)
                pos[1] = random.randint(0, self.size-1)
            occupied.append(pos)

        # Flags: 0=not collected, 1=collected
        self.coin1_flag = 0
        self.coin2_flag = 0
        self.key_flag = 0

        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return tuple(self.agent_pos + [self.coin1_flag, self.coin2_flag, self.key_flag])

    def step(self, action):
        self.steps += 1

        # Move agent
        if action == 0 and self.agent_pos[1] > 0:  # up
            self.agent_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < self.size-1:  # down
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:  # left
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.size-1:  # right
            self.agent_pos[0] += 1

        reward = 0

        # Check Coin1
        if self.agent_pos == self.coin1_pos and self.coin1_flag == 0:
            reward += 1
            self.coin1_flag = 1

        # Check Coin2
        if self.agent_pos == self.coin2_pos and self.coin2_flag == 0:
            reward += 1
            self.coin2_flag = 1

        # Check Key
        if self.agent_pos == self.key_pos and self.key_flag == 0:
            reward += 2
            self.key_flag = 1

        # Check Exit
        done = False
        if self.agent_pos == self.exit_pos:
            if self.key_flag == 1:
                reward += 5
                done = True
            else:
                reward += 0  # cannot exit without key

        # Episode ends if max steps reached
        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def render(self):
        grid = np.full((self.size, self.size), '.')
        grid[self.coin1_pos[1], self.coin1_pos[0]] = 'C' if self.coin1_flag == 0 else '.'
        grid[self.coin2_pos[1], self.coin2_pos[0]] = 'C' if self.coin2_flag == 0 else '.'
        grid[self.key_pos[1], self.key_pos[0]] = 'K' if self.key_flag == 0 else '.'
        grid[self.exit_pos[1], self.exit_pos[0]] = 'E'
        grid[self.agent_pos[1], self.agent_pos[0]] = 'A'
        print('\n'.join([' '.join(row) for row in grid]))
        print(f"Flags: coin1={self.coin1_flag}, coin2={self.coin2_flag}, key={self.key_flag}")
        print()

# ---------------------------
# 2. Tabular Q-Learning Agent
# ---------------------------
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.2):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table shape: agent_x, agent_y, coin1_flag, coin2_flag, key_flag, action
        size = env.size
        self.Q = np.zeros((size, size, 2, 2, 2, env.action_space.n))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        best_next = np.max(self.Q[next_state]) if not done else 0
        self.Q[state + (action,)] += self.alpha * (reward + self.gamma * best_next - self.Q[state + (action,)])

# ---------------------------
# 3. Training Loop
# ---------------------------
env = MultiGoalGridEnv()
agent = QLearningAgent(env)

num_episodes = 1000

for ep in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if (ep+1) % 100 == 0:
        print(f"Episode {ep+1}, Total Reward: {total_reward}")

# ---------------------------
# 4. Testing / Rendering
# ---------------------------
print("\nTesting trained agent:\n")
state = env.reset()
done = False
env.render()

while not done:
    action = np.argmax(agent.Q[state])
    state, reward, done, _ = env.step(action)
    env.render()
