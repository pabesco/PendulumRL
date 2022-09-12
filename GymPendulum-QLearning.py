import gym
import numpy as np
import random

env = gym.make("Pendulum-v0")
#env = gym.make("MountainCar-v0")


LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 24100
SHOW_EVERY = 3000

epsilon = 0.2
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = 25000
epsilon_decay = epsilon/(END_EPSILON_DECAY - START_EPSILON_DECAY)


DISCRETE_OS_SIZE = [20]*len(env.observation_space.high)
#print(DISCRETE_OS_SIZE)
discrete_os_win_size = (env.observation_space.high+1 - env.observation_space.low)/DISCRETE_OS_SIZE

ACTION_CHOICES = 1000

action_space = np.linspace(int(env.action_space.low), int(env.action_space.high), ACTION_CHOICES)

#print(action_space)

'''
q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [len(action_space)]))
'''
q_table = np.load(f"qtables_pendulum/8_{24000}-qtable.npy")
#print(q_table.shape)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(int))

count = 0

for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    print(episode)
    render = False
    if episode % SHOW_EVERY == 0:
        render = True
        np.save(f"qtables_pendulum/9_{episode}-qtable.npy", q_table)
        print(f"Success Rate ==> {count*100/(episode+1)}")

    done = False

    while not done:

        if np.random.random() > epsilon:
            action = [action_space[np.argmax(q_table[discrete_state])]]
            action_index = np.argmax(q_table[discrete_state])
        else:
            action = [random.choice(action_space)]
            action_index = np.where(action_space == action[0])

        #print(action)
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action_index,)]
            new_q = current_q*(1 - LEARNING_RATE) + LEARNING_RATE*(reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action_index, )] = new_q
        elif reward >= -0.1:
            count += 1
            #print(f"We made it on episode {episode} ==> {count*100/(episode+1)}")
            q_table[discrete_state + (action_index, )] = 0

        discrete_state = new_discrete_state

        if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAY:
            epsilon -= epsilon_decay

env.close()
