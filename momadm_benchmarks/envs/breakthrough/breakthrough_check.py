"""MO Breakthrough check."""

from breakthrough import MOBreakthrough


def check_env():
    """Check the environment."""
    environment = MOBreakthrough(board_width=6, board_height=6, num_objectives=4)
    environment.reset(seed=42)

    for agent in environment.agent_iter():
        observation, reward, termination, truncation, info = environment.last()

        print("rewards", environment.rewards)
        if termination or truncation:
            action = None
        else:
            if observation:
                mask = observation["action_mask"]
                # this is where you would insert your policy
                action = environment.action_space(agent).sample(mask)
                # print("observation: ", observation)
                print("cumulative rewards", environment._cumulative_rewards)
                print("action: ", action)

        environment.step(action)

    # print("observation: ", observation)
    # print("reward: ", reward)
    print("rewards", environment.rewards)
    print("cumulative rewards", environment._cumulative_rewards)
    # name = input("Press key to end\n")
    environment.close()


if __name__ == "__main__":
    (check_env())
