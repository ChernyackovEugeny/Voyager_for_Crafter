from environment.wrapper import CrafterEnv
from environment.captioner import caption
from agent.agent import Agent


env = CrafterEnv()
agent = Agent(env)
agent.run()
