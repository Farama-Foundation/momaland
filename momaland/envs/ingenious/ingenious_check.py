"""Temporary check file for Ingenious environment."""

from ingenious import MOIngenious


def train():
    """Train a random agent on the item gathering domain."""
    done = False

    while not done:
        ag = ig_env.agent_selection
        print("Agent: ", ag)
        action = ig_env.action_space(ag).sample()
        print("Action: ", action)
        ig_env.step(action)
        observation, reward, truncation, termination, _ = ig_env.last()
        print("Observations: ", observation)
        print("Rewards: ", reward)
        print("Truncation: ", truncation)
        print("Termination: ", termination)
        done = truncation or termination


if __name__ == "__main__":
    ig_env = MOIngenious(num_players=2, init_draw=6, num_colors=6, board_size=8)

    # print(ig_env.reset())

    train()
