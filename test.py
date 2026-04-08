import gymnasium as gym
from shimmy import GymV21CompatibilityV0
import crafter

env = GymV21CompatibilityV0(env_id='CrafterReward-v1')
obs, info = env.reset()

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print('+++++++++++++++++++++++++++')
print(obs)
print(info)
input()