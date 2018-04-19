import logging
# from gym.envs.registration import register

# Multiagent envs
# ----------------------------------------

# register(
#     id='MultiagentSimple-v0',
#     entry_point='multiagent.envs:SimpleEnv',
#     # FIXME(cathywu) currently has to be exactly max_path_length parameters in
#     # rllab run script
#     max_episode_steps=100,
# )

# register(
#     id='MultiagentSimpleSpeakerListener-v0',
#     entry_point='multiagent.envs:SimpleSpeakerListenerEnv',
#     max_episode_steps=100,
# )

logger_agent = logging.getLogger('GridMARL')
logger_agent.setLevel(logging.INFO)

fm = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > [%(name)s] %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(fm)
logger_agent.addHandler(sh)

# fh_agent = logging.FileHandler('./agent.log')
# fh_agent.setFormatter(fm)
# logger_agent.addHandler(fh_agent)
