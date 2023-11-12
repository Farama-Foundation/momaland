"""MO Connect4 check."""

from connect4 import MOConnect4


def main():
    """Check the environment."""
    environment = MOConnect4(board_width=7, board_height=6, column_objectives=False)
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
                # print("cumulative rewards", environment._cumulative_rewards)
                # print("action: ", action)

        environment.step(action)

    # print("observation: ", observation)
    print("reward: ", reward)
    print("rewards", environment.rewards)
    # print("cumulative rewards", environment._cumulative_rewards)
    # name = input("Press key to end\n")
    environment.close()


if __name__ == "__main__":
    main()
