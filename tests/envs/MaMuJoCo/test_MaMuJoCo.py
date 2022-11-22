from pettingzoo.test import parallel_api_test
from multiagent_mujoco.mujoco_multi import MaMuJoCo

if __name__ == "__main__":
    for ok in [None, 0, 1]:
        scenario = "InvertedPendulum"  # for debugging
        agent_conf = None
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "Ant"
        agent_conf = None
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "Ant"
        agent_conf = "2x4"
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "Ant"
        agent_conf = "2x4d"
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "Ant"
        agent_conf = "4x2"
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "HalfCheetah"
        agent_conf = "2x3"
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "HalfCheetah"
        agent_conf = "6x1"
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "HalfCheetah"
        agent_conf = None
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "Hopper"
        agent_conf = "3x1"
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "Hopper"
        agent_conf = None
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "Humanoid"
        agent_conf = "9|8"
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "Humanoid"
        agent_conf = None
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "HumanoidStandup"
        agent_conf = "9|8"
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "HumanoidStandup"
        agent_conf = None
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "Reacher"
        agent_conf = "2x1"
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "Reacher"
        agent_conf = None
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "Swimmer"
        agent_conf = "2x1"
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "Swimmer"
        agent_conf = None
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "Walker2d"
        agent_conf = "2x3"
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "Walker2d"
        agent_conf = None
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "manyagent_swimmer"
        agent_conf = "10x2"
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)
        agent_conf = "6x1"
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "manyagent_ant"
        agent_conf = "2x3"
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)
        agent_conf = "3x1"
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "coupled_half_cheetah"
        agent_conf = "1p1"
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario = "coupled_half_cheetah"
        agent_conf = None
        parallel_api_test(MaMuJoCo(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)
