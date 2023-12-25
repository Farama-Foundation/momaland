"""MO Breakthrough check."""

import numpy as np
from breakthrough import MOBreakthrough


def check_env():
    """Check the environment."""
    environment = MOBreakthrough(board_width=6, board_height=6, num_objectives=4, render_mode="ansi")
    environment.reset(seed=42)

    for agent in environment.agent_iter():
        observation, reward, termination, truncation, info = environment.last()

        print("rewards from the last timestep", reward)
        print("cumulative rewards before action", environment._cumulative_rewards)
        if termination or truncation:
            action = None
        else:
            if observation:
                print("environment before action:")
                environment.render()
                # print("observation:")
                # print(observation)
                # this is where you would insert your policy
                action = np.where(observation["action_mask"] != 0)[0][0]
                print("action: ", action)

        environment.step(action)
        print("environment after action:")
        environment.render()
        print("cumulative rewards after action", environment._cumulative_rewards)

    # print("observation: ", observation)
    # print("reward: ", reward)
    print("game end: rewards", environment.rewards)
    print("game end: cumulative rewards", environment._cumulative_rewards)
    # name = input("Press key to end\n")
    environment.close()


if __name__ == "__main__":
    (check_env())
