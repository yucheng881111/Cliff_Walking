import numpy as np
from Env import CliffWalking
import matplotlib.pyplot as plt

# action: 0-up, 1-down, 2-left, 3-right

def epsilon_greedy_policy(state, q_table, epsilon=0.1):
    decide_explore_exploit  = np.random.random()
    
    if decide_explore_exploit < epsilon:
        action = np.random.choice(4)
    else:
        action = np.argmax(q_table[:, state]) # Choose the action with largest Q-value (state value)
        
    return action


def update_q_table(q_table, state, action, reward, next_state_value, gamma_discount=0.9, alpha=0.5):
    """
    Update the q_table based on observed rewards and next state value
    Q(S, A) <- Q(S, A) + [ alpha * (reward + (gamma * maxQ(S', A')) - Q(S, A) ]
    """

    update_q_value = q_table[action, state] + alpha * (reward + (gamma_discount * next_state_value) - q_table[action, state])
    q_table[action, state] = update_q_value

    return q_table  

def tabular_q_learning(num_episodes=1000, gamma_discount=0.9, alpha=0.5, epsilon=0.1):
    # initialize all states to 0
    env = CliffWalking()
    q_table = np.zeros((4, env.size[0] * env.size[1])) # 4 actions, 48 states
    total_reward_list = []
    total_step_list = []
    # training
    for episode in range(0, num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        total_step = 0
        while not done:
            # choose action using epsilon-greedy policy
            action = epsilon_greedy_policy(state, q_table, epsilon)
            next_state, reward, done = env.step(action)
            total_step += 1
            # get maxValue(Q(S', A'))
            all_next_state_value = q_table[:, next_state]
            maximum_next_state_value = np.amax(all_next_state_value)

            total_reward += reward 
            # update q_table
            q_table = update_q_table(q_table, state, action, reward, maximum_next_state_value, gamma_discount, alpha)
            # update the state
            state = next_state

        total_reward_list.append(total_reward)
        total_step_list.append(total_step)
        print("Episode: {}, total reward: {}, total step: {}".format(episode+1, total_reward, total_step))
    
    # evaluating
    state = env.reset()
    done = False
    total_reward = 0
    action_list = []
    while True:
        env.render()
        action = np.argmax(q_table[:, state])
        next_state, reward, done = env.step(action)
        action_list.append(action)
        total_reward += reward
        if done:
            env.render()
            break

        state = next_state
    
    print("total reward: ", total_reward)

    return q_table, total_reward_list, total_step_list, action_list

if __name__ == "__main__":
    q_table, total_reward_list, total_step_list, action_list = tabular_q_learning()
    plt.plot(total_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()
