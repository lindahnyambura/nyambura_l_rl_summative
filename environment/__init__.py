## env registry
from gymnasium.envs.registration import register
register(
    id="NairobiProtestEnv-v0",
    entry_point="environment.custom_env:NairobiCBDProtestEnv",
    max_episode_steps=1000,
)