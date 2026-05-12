import crafter

# crafter uses old gym API: reset() -> obs, step() -> (obs, reward, done, info)
raw_env = crafter.Env()

obs = raw_env.reset()
action = raw_env.action_space.sample()
obs, reward, done, info = raw_env.step(action)

# adapt to gymnasium-style (terminated/truncated)
terminated = bool(done)
truncated = False

print('+++++++++++++++++++++++++++')
print(obs)
print(info)
input()
