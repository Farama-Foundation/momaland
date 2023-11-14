"""Requirement for users.

- The package `ffmpeg` is required to be installed on host system for the PNG[] -> GIF process.
"""

import sys

from PIL import Image

from momaland.utils.all_modules import all_environments


def generate_gif(nameline, module):
    """Generates a GIF of a full environment cycle."""
    env = module.env(render_mode="rgb_array")
    env.reset()
    imgs = []
    for _ in range(100):
        for agent in env.agent_iter(env.num_agents):  # step through every agent once with observe=True
            obs, rew, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                action = env.action_spaces[agent].sample()
            env.step(action)

        # save rgb_array data
        ndarray = env.render()
        im = Image.fromarray(ndarray)
        imgs.append(im)

    env.close()

    # render gif from data
    imgs[0].save(f"{nameline}.gif", save_all=True, append_images=imgs[1:], duration=40, loop=0)


if __name__ == "__main__":
    name = sys.argv[1]
    if name == "all":
        for name, module in all_environments.items():
            nameline = name.replace("/", "_")
            generate_gif(nameline, module)
    else:
        module = all_environments[name]
        nameline = name.replace("/", "_")
        generate_gif(nameline, module)
