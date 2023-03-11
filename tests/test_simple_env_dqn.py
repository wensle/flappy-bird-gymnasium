import time

import gymnasium
import numpy as np
import pygame
import tensorflow as tf

import flappy_bird_gymnasium

env = gymnasium.make("FlappyBird-v0")


class DuelingDQN(tf.keras.Model):
    def __init__(self, action_space):
        super(DuelingDQN, self).__init__()

        self.fc1 = tf.keras.layers.Dense(
            512,
            activation="elu",
            kernel_initializer=tf.keras.initializers.Orthogonal(tf.sqrt(2.0)),
        )
        self.fc2 = tf.keras.layers.Dense(
            256,
            activation="elu",
            kernel_initializer=tf.keras.initializers.Orthogonal(tf.sqrt(2.0)),
        )
        self.V = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
        )
        self.A = tf.keras.layers.Dense(
            action_space,
            activation=None,
            kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
        )

    def call(self, inputs, training=None):
        x = self.fc1(inputs, training=training)
        x = self.fc2(x, training=training)
        V = self.V(x, training=training)
        A = self.A(x, training=training)
        adv_mean = tf.reduce_mean(A, axis=-1, keepdims=True)
        return V + (A - adv_mean)

    def get_action(self, state):
        q_value = self(state)
        print("Q value: ", q_value, tf.math.argmax(q_value, axis=-1))
        return tf.math.argmax(q_value, axis=-1)[0]


# init models
q_model = DuelingDQN(env.action_space.n)
q_model.build((None, env.observation_space.shape[0]))
q_model.load_weights("model.h5")

# run
for epoch in range(100):
    clock = pygame.time.Clock()
    score = 0

    state, _ = env.reset()
    state = np.expand_dims(state, axis=0)
    while True:
        env.render()

        # Getting action
        action = q_model.get_action(state)
        action = np.array(action, copy=False, dtype=env.env.action_space.dtype)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Processing action
        next_state, reward, done, _, info = env.step(action)

        state = np.expand_dims(next_state, axis=0)
        score += reward
        print(f"Obs: {state}\n" f"Action: {action}\n" f"Score: {score}\n")

        clock.tick(30)

        if done:
            env.render()
            time.sleep(0.6)
            break

env.close()
