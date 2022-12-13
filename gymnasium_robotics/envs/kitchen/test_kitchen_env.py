from kitchen_env import FrankaRobot

env = FrankaRobot()
env.reset()
env.ik_controller.get_params()