import gymnasium as gym 
import numpy as np
import matplotlib.pyplot as plt
import pickle

episode_count = 100
render = False
running = True

env = gym.make('MountainCar-v0', render_mode='human' if render else None)

position_segments = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
velocity_segments = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

if running:
    q_table = np.zeros((len(position_segments), len(velocity_segments), env.action_space.n))
else:
    f = open("q_table.pkl", "rb")
    q_table = pickle.load(f)
    f.close()

learning_rate = 0.9
discount_factor = 0.9

epsilon = 1
epsilon_decay = 2/episode_count
random = np.random.default_rng()

rewards_per_episode = np.zeros(episode_count)

for episode in range(episode_count):
    state = env.reset()[0]
    state_position = np.digitize(state[0], position_segments)
    state_velocity = np.digitize(state[1], velocity_segments)

    terminated = False

    rewards = 0

    while not terminated and rewards > -1000:
        if running and random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_position][state_velocity])
        
        new_state, reward, terminated, _, _ = env.step(action)
        new_state_position = np.digitize(new_state[0], position_segments)
        new_state_velocity = np.digitize(new_state[1], velocity_segments)
        
        if running:
            q_table[state_position][state_velocity][action] += learning_rate * (reward + 
                                                                                discount_factor * 
                                                                                np.max(q_table[new_state_position][new_state_velocity]) - 
                                                                                q_table[state_position][state_velocity][action])
        
        state = new_state
        state_position = new_state_position
        state_velocity = new_state_velocity
        rewards += reward
    
    epsilon = max(epsilon - epsilon_decay, 0)
    rewards_per_episode[episode] = rewards
    
env.close()

if running:
    f = open("q_table.pkl", "wb")
    pickle.dump(q_table, f)
    f.close()

    mean_rewards = np.zeros(episode_count)
    for i in range(episode_count):
        mean_rewards[i] = np.mean(rewards_per_episode[max(0, i-100):i+1])
    plt.plot(mean_rewards)
    plt.savefig('rewards.png')