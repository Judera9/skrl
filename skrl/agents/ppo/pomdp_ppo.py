from typing import Any, Mapping, Optional, Tuple, Union, Dict

import copy
import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents import Agent
from skrl.memories import Memory
from skrl.models import Model
from skrl.resources.schedulers import KLAdaptiveLR
from skrl.resources.schedulers import ConstantScheduler

# fmt: off
# [start-config-dict-torch]
POMDP_PPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})
    "entropy_scheduler": None,        # entropy scheduler class (see skrl.resources.schedulers)
    "entropy_scheduler_kwargs": {},   # entropy scheduler's kwargs (e.g. {"step_size": 1e-3})

    "use_encoder": False,                # encoder class (see skrl.models)
    "encoder_kwargs": {},           # encoder's kwargs (e.g. {"name": "encoder"})
    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "actor_observation_preprocessor": None, # actor observation preprocessor class (see skrl.resources.preprocessors)
    "actor_observation_preprocessor_kwargs": {}, # actor observation preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,        # entropy loss scale (default: 0.0)
    "grad_penalty_weight": 0.0,        # gradient penalty weight for Lipschitz continuity (default: 0.0)
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


class POMDP_PPO(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        actor_observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Proximal Policy Optimization (PPO)

        https://arxiv.org/abs/1707.06347

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.Memory, list of skrl.memory.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param actor_observation_space: Actor observation/state space or shape (default: ``None``)
        :type actor_observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(POMDP_PPO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )
        self.actor_observation_space = actor_observation_space

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
                if self.value is not None and self.policy is not self.value:
                    self.value.broadcast_parameters()

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]
        self._grad_penalty_weight = self.cfg["grad_penalty_weight"]

        self._kl_threshold = self.cfg["kl_threshold"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]
        self._entropy_scheduler = self.cfg["entropy_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._actor_observation_preprocessor = self.cfg["actor_observation_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # configure encoders
        self._use_encoder = self.cfg["use_encoder"]
        self._encoder_kwargs = self.cfg["encoder_kwargs"]
        if self._use_encoder:
            self.encoder = self.models.get(self._encoder_kwargs["model_name"], None)
            self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self._encoder_kwargs["learning_rate"])
            self.checkpoint_modules["encoder"] = self.encoder
            self.checkpoint_modules["encoder_optimizer"] = self.encoder_optimizer

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
            else:
                self.optimizer = torch.optim.Adam(
                    itertools.chain(self.policy.parameters(), self.value.parameters()), lr=self._learning_rate
                )
            if self._learning_rate_scheduler is not None:
                self.learning_rate_scheduler = self._learning_rate_scheduler(
                    self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )
            if self._entropy_scheduler is not None:
                self.entropy_scheduler = self._entropy_scheduler(
                    self._entropy_loss_scale, **self.cfg["entropy_scheduler_kwargs"]
                )

            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        if self._actor_observation_preprocessor:
            self._actor_observation_preprocessor = self._actor_observation_preprocessor(**self.cfg["actor_observation_preprocessor_kwargs"])
            self.checkpoint_modules["actor_observation_preprocessor"] = self._actor_observation_preprocessor
        else:
            self._actor_observation_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

        if self._use_encoder:
            self._encoder_preprocessor = self._encoder_kwargs["encoder_preprocessor"]
            if self._encoder_preprocessor:
                self._encoder_preprocessor = self._encoder_preprocessor(**self._encoder_kwargs["encoder_preprocessor_kwargs"])
                self.checkpoint_modules["encoder_preprocessor"] = self._encoder_preprocessor
            else:
                self._encoder_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actor_observations", size=self.actor_observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

            # tensors sampled during training
            self._tensors_names = ["states", "actor_observations", "actions", "log_prob", "values", "returns", "advantages"]

            if self._use_encoder:
                self.other_observation_space = self.encoder.action_space
                self.memory.create_tensor(name=f"{self._encoder_kwargs['obs_name']}", size=self.other_observation_space, dtype=torch.float32)
                self._tensors_names.append(f"{self._encoder_kwargs['obs_name']}")

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None
        self._current_next_actor_observations = None

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        if self._use_encoder:
            self.encoder.set_mode("eval")
            states = self._actor_observation_preprocessor(states)
            raw_encoder_output, _, _ = self.encoder.compute({"states": states}, role="policy")
            encoder_output = self._encoder_preprocessor(raw_encoder_output, train=False)
            states = torch.cat([states, encoder_output], dim=-1)
        else:
            states = self._actor_observation_preprocessor(states)

        # sample random actions
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": states}, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, log_prob, outputs = self.policy.act({"states": states}, role="policy")
            self._current_log_prob = log_prob

        return actions, log_prob, outputs

    def act_eval(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy, without exploration

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        if self._use_encoder:
            self.encoder.set_mode("eval")
            states = self._actor_observation_preprocessor(states)
            raw_encoder_output, _, _ = self.encoder.compute({"states": states}, role="policy")
            encoder_output = self._encoder_preprocessor(raw_encoder_output, train=False)
            states = torch.cat([states, encoder_output], dim=-1)
        else:
            states = self._actor_observation_preprocessor(states)

        # sample deterministic actions
        actions, _, outputs = self.policy.compute({"states": states}, role="policy")

        return actions, None, outputs

    def record_transition(
        self,
        states: torch.Tensor,
        actor_observations: torch.Tensor,
        other_observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        next_actor_observations: torch.Tensor,
        next_other_observations: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param next_actor_observations: Next actor observations of the environment
        :type next_actor_observations: torch.Tensor
        :param next_other_observations: Next other observations of the environment
        :type next_other_observations: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            self._current_next_states = next_states
            self._current_next_actor_observations = next_actor_observations

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values
            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                values, _, _ = self.value.act({"states": self._state_preprocessor(states)}, role="value")
                values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) bootstrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # storage transition in memory
            memory_samples = {
                "states": states,
                "actor_observations": actor_observations,
                "actions": actions,
                "rewards": rewards,
                "next_states": next_states,
                "next_actor_observations": next_actor_observations,
                "terminated": terminated,
                "truncated": truncated,
                "log_prob": self._current_log_prob,
                "values": values,
            }
            if self._use_encoder:
                memory_samples[f"{self._encoder_kwargs['obs_name']}"] = other_observations[f"{self._encoder_kwargs['obs_name']}"]
            self.memory.add_samples(**memory_samples)
            for memory in self.secondary_memories:
                memory.add_samples(**memory_samples)
            for name, item in infos["log"].items():
                if "Metrics" in name or "Episode_Termination" in name:
                    self.track_data(name, item)

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _calc_grad_penalty(self, obs_batch, actions_log_prob_batch):
        grad_log_prob = torch.autograd.grad(actions_log_prob_batch.sum(), obs_batch, create_graph=True)[0]
        gradient_penalty_loss = torch.sum(torch.square(grad_log_prob), dim=-1).mean()
        return gradient_penalty_loss

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value"
            )
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_grad_penalty = 0
        cumulative_encoder_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for batch in sampled_batches:
                if self._use_encoder:
                    sampled_states, sampled_actor_observations, sampled_actions, sampled_log_prob, sampled_values, sampled_returns, sampled_advantages, sampled_vel_est = batch
                else:
                    sampled_states, sampled_actor_observations, sampled_actions, sampled_log_prob, sampled_values, sampled_returns, sampled_advantages = batch

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                    sampled_states = self._state_preprocessor(sampled_states, train=not epoch)
                    sampled_actor_observations = self._actor_observation_preprocessor(sampled_actor_observations, train=not epoch)

                    # Enable gradient computation for gradient penalty
                    if self._grad_penalty_weight > 0:
                        sampled_actor_observations.requires_grad_(True)

                    if self._use_encoder:
                        self.encoder.set_mode("train")
                        raw_encoder_output, _, _ = self.encoder.compute({"states": sampled_actor_observations}, role="policy")
                        encoder_output = self._encoder_preprocessor(raw_encoder_output, train=not epoch)
                        sampled_actor_observations = torch.cat([sampled_actor_observations, encoder_output.detach()], dim=-1)

                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_actor_observations, "taken_actions": sampled_actions}, role="policy"
                    )

                    # compute gradient penalty for Lipschitz continuity
                    if self._grad_penalty_weight > 0:
                        gradient_penalty_loss = self._calc_grad_penalty(sampled_actor_observations, next_log_prob)
                        gradient_penalty_loss = self._grad_penalty_weight * gradient_penalty_loss
                    else:
                        gradient_penalty_loss = 0

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    # compute entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.entropy.mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    encoder_loss = self._update_encoder(raw_encoder_output, sampled_vel_est)

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss + gradient_penalty_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._grad_penalty_weight > 0:
                    cumulative_grad_penalty += gradient_penalty_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                if self._use_encoder:
                    cumulative_encoder_loss += encoder_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.learning_rate_scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.learning_rate_scheduler.step(kl.item())
                else:
                    self.learning_rate_scheduler.step()

        # update entropy scheduler
        if self._entropy_scheduler:
            self.entropy_scheduler.step()
            self._entropy_loss_scale = self.entropy_scheduler.get_value()

        # record data
        self.track_data("Loss/surrogate_loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss/value_loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss/kl_divergence", torch.tensor(kl_divergences, device=self.device).mean().item())
        self.track_data("Loss/entropy", self.policy.entropy.mean().item())
        if self._grad_penalty_weight > 0:
            self.track_data("Loss/grad_penalty", cumulative_grad_penalty / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data("Loss/entropy_loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches))
            if self._entropy_scheduler:
                self.track_data("Loss/entropy_loss_scale", self._entropy_loss_scale)
        self.track_data("Policy/mean_noise_std", self.policy.action_std.mean().item())
        if self._learning_rate_scheduler:
            self.track_data("Loss/learning_rate", self.learning_rate_scheduler.get_last_lr()[0])
        if self.policy._noise_generator and type(self.policy.noise_generator).__name__ == "PinkNoiseDist":
            self.track_data("Loss/noise_smoothing", self.policy.noise_generator.smoothing)
        if self._use_encoder:
            self.track_data(f"Loss/{self._encoder_kwargs['model_name']}_loss", cumulative_encoder_loss / (self._learning_epochs * self._mini_batches))

    def _update_encoder(self, estimates, targets, loss="mse") -> float:
        # compute encoder loss
        encoder_loss = 0
        
        # Compute loss
        if loss == "mse":
            encoder_loss = F.mse_loss(estimates, targets)
        else:
            raise ValueError(f"Loss {loss} not supported")  
        
        # Update encoder separately
        self.encoder_optimizer.zero_grad()
        self.scaler.scale(encoder_loss).backward()
        
        if self._grad_norm_clip > 0:
            self.scaler.unscale_(self.encoder_optimizer)
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self._grad_norm_clip)
        
        self.scaler.step(self.encoder_optimizer)
        self.encoder.set_mode("eval")
        return encoder_loss