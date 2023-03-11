# MIT License
#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
# Copyright (c) 2023 Martin Kubovcik
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" Handles the initialization of the game through the command line interface.
"""

import argparse
import time
import pygame

import gymnasium

import flappy_bird_gymnasium


def _get_args():
    """Parses the command line arguments and returns them."""
    parser = argparse.ArgumentParser(description=__doc__)

    # Argument for the mode of execution (human or random):
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="human",
        choices=["human", "random"],
        help="The execution mode for the game.",
    )

    return parser.parse_args()

def human_agent_env():
    env = gymnasium.make("FlappyBird-v0")

    clock = pygame.time.Clock()
    score = 0

    obs = env.reset()
    while True:
        env.render()

        # Getting action:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN and (
                event.key == pygame.K_SPACE or event.key == pygame.K_UP
            ):
                action = 1

        # Processing:
        obs, reward, done, _, info = env.step(action)

        score += reward
        print(f"Obs: {obs}\n" f"Action: {action}\n" f"Score: {score}\n")

        clock.tick(15)

        if done:
            env.render()
            time.sleep(0.6)
            break

    env.close()

def random_agent_env():
    env = gymnasium.make("FlappyBird-v0")
    env.reset()
    score = 0
    while True:
        env.render()

        # Getting random action:
        action = env.action_space.sample()

        # Processing:
        obs, reward, done, _, _ = env.step(action)

        score += reward
        print(f"Obs: {obs}\n" f"Action: {action}\n" f"Score: {score}\n")

        time.sleep(1 / 30)

        if done:
            env.render()
            time.sleep(0.5)
            break


def main():
    args = _get_args()

    if args.mode == "human":
        human_agent_env()
    elif args.mode == "random":
        random_agent_env()
    else:
        print("Invalid mode!")
