import numpy as np

class CliffWalking:
    def __init__(self, size=(4, 12), start=(3, 0), end=(3, 11), obstacles=[]):
        self.size = size
        self.start = start
        self.end = end
        self.state = start
        self.obstacles = obstacles
        self.length = 0
        self.max_length = 100

    def reset(self):
        self.length = 0
        self.state = self.start
        return self.state_encode(self.state)

    def state_encode(self, state):
        return state[0] * self.size[1] + state[1]

    def get_reward(self, state):
        if state == self.end:
            return 10
        elif self.state_encode(state) >= 37 and self.state_encode(state) <= 46:
            return -100
        else:
            return -1

    def step(self, action):
        if action == 0:    # up
            next_state = (self.state[0] - 1, self.state[1])
        elif action == 1:  # down
            next_state = (self.state[0] + 1, self.state[1])
        elif action == 2:  # left
            next_state = (self.state[0], self.state[1] - 1)
        elif action == 3:  # right
            next_state = (self.state[0], self.state[1] + 1)
        else:
            raise ValueError("Invalid action.")

        if next_state[0] < 0 or next_state[0] >= self.size[0] or next_state[1] < 0 or next_state[1] >= self.size[1]:
            next_state = self.state
        elif next_state in self.obstacles:
            next_state = self.state
        
        self.length += 1
        reward = self.get_reward(next_state)
        done = False
        if next_state == self.end or self.length == self.max_length or (self.state_encode(next_state) >= 37 and self.state_encode(next_state) <= 46):
            done = True

        self.state = next_state
        return self.state_encode(next_state), reward, done
    
    def sample(self):
        return np.random.randint(0, 4)

    def render(self):
        print()
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if (i, j) == self.state:
                    print("S", end=" ")
                elif (i, j) == self.end:
                    print("E", end=" ")
                elif (i, j) in self.obstacles:
                    print("O", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()

if __name__ == '__main__':
    env = CliffWalking()
    state = env.reset()
    total_reward = 0
    action_list = [0,3,3,3,3,3,3,3,3,3,3,3,1]
    while True:
        env.render()
        action = action_list.pop(0)
        print("action: ", action)
        next_state, reward, done = env.step(action)
        print("reward: ", reward)

        total_reward += reward
        print(state, "->", next_state)
        if done:
            env.render()
            break

        state = next_state
    
    print("total reward: ", total_reward)