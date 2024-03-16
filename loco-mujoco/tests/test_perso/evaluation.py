from mushroom_rl.core import Core, Agent
from loco_mujoco import LocoEnv


env = LocoEnv.make("UnitreeA1.simple")

agent = Agent.load("/home/arthur/Documents/Imitation_learning/loco-mujoco/tests/test_perso/logs/loco_mujoco_evaluation_2024-03-16_12-58-33/env_id___UnitreeA1.simple.real/0/agent_epoch_156_J_42.468310.msh")

core = Core(agent, env)

core.evaluate(n_episodes=10, render=True)