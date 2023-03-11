
import flappy_bird_gymnasium
import gymnasium
import numpy as np

class TestFlappyBird:
    def test_one(self):
        env = gymnasium.make("FlappyBird-v0")

        obs, _ = env.reset()
        score = 0
        while True:
            # Next action:
            # (feed the observation to your agent here)
            action = env.action_space.sample()

            # Processing:
            obs, reward, terminated, _, info = env.step(action)
            
            score += reward

            # Checking if the player is still alive
            if terminated:
                break
        
        env.close()
        assert obs.shape == (9,)
        assert action.shape == (2,)
        assert info["score"] == 0
        np.testing.assert_allclose(score, 8.99999999999998)