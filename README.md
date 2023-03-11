## Flappy Bird for Gymnasium

![Python versions](https://img.shields.io/pypi/pyversions/flappy-bird-gymnasium)
[![PyPI](https://img.shields.io/pypi/v/flappy-bird-gymnasium)](https://pypi.org/project/flappy-bird-gymnasium/)
[![License](https://img.shields.io/github/license/markub3327/flappy-bird-gymnasium)](https://github.com/markub3327/flappy-bird-gymnasium/blob/master/LICENSE)

This repository contains the implementation of two OpenAI Gym environments for
the Flappy Bird game. The implementation of the game's logic and graphics was
based on the [FlapPyBird](https://github.com/sourabhv/FlapPyBird) project, by
[@sourabhv](https://github.com/sourabhv). 

The two environments differ only on the type of observations they yield for the
agents. The "FlappyBird-rgb-v0" environment, yields RGB-arrays (images)
representing the game's screen. The "FlappyBird-v0" environment, on the other
hand, yields simple numerical information about the game's state as
observations. The yielded attributes are the:

* horizontal distance to the next pipe
* difference between the player's y position and the next hole's y position
* horizontal distance to the next next pipe
* difference between the player's y position and the next next hole's y position
* player's vertical velocity
* player's rotation

<br>

<p align="center">
  <img align="center" 
       src="https://github.com/markub3327/flappy-bird-gymnasium/blob/main/imgs/yellow_bird_playing.gif?raw=true" 
       width="200"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img align="center" 
       src="https://github.com/markub3327/flappy-bird-gymnasium/blob/main/imgs/red_bird_start_screen.gif?raw=true" 
       width="200"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img align="center" 
       src="https://github.com/markub3327/flappy-bird-gymnasium/blob/main/imgs/blue_bird_playing.gif?raw=true" 
       width="200"/>
</p>

## Installation

To install `flappy-bird-gymnasium`, simply run the following command:

    $ pip install flappy-bird-gymnasium
    
## Usage

Like with other `gymnasium` environments, it's very easy to use `flappy-bird-gymnasium`.
Simply import the package and create the environment with the `make` function.
Take a look at the sample code below:

```
import time
import flappy_bird_gymnasium
import gymnasium
env = gymnasium.make("FlappyBird-v0")

obs, _ = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()

    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    
    # Rendering the game:
    # (remove this two lines during training)
    env.render()
    time.sleep(1 / 30)  # FPS
    
    # Checking if the player is still alive
    if terminated:
        break

env.close()
```

## Playing

To play the game (human mode), run the following command:

    $ flappy_bird_gymnasium
    
To see a random agent playing, add an argument to the command:

    $ flappy_bird_gymnasium --mode random
