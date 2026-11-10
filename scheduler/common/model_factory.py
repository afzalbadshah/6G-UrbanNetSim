from scheduler.models.a2c_policy import A2CPolicy
from scheduler.models.dqn_policy import DQNPolicy
from scheduler.models.double_dqn_policy import DoubleDQNPolicy
from scheduler.models.ppo_policy import PPOPolicy
from scheduler.models.q_learning_policy import QLearningPolicy
from scheduler.models.random_policy import RandomPolicy
from scheduler.models.rule_based_policy import RuleBasedPolicy


def create_policy(model_name, state_dim, action_dim, config):
    name = str(model_name or "").strip().lower()
    if name == "random":
        return RandomPolicy(state_dim, action_dim, config)
    if name == "rule_based":
        return RuleBasedPolicy(state_dim, action_dim, config)
    if name == "q_learning":
        return QLearningPolicy(state_dim, action_dim, config)
    if name == "dqn":
        return DQNPolicy(state_dim, action_dim, config)
    if name == "double_dqn":
        return DoubleDQNPolicy(state_dim, action_dim, config)
    if name == "ppo":
        return PPOPolicy(state_dim, action_dim, config)
    if name == "a2c":
        return A2CPolicy(state_dim, action_dim, config)
    raise ValueError(f"Unknown SmartCity comparison model '{model_name}'.")
