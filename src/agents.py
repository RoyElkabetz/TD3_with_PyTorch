import os
import torch as T
import torch.nn.functional as F
import numpy as np

from buffer import ReplayBuffer
from networks import Actor, Critic


class Agent:
    def __init__(self, gamma, alpha, beta, state_dims, action_dims, max_action, min_action, fc1_dim, fc2_dim,
                 memory_size, batch_size, tau, update_period, noise_std, noise_clip, warmup, name, ckpt_dir='tmp'):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.max_action = max_action
        self.min_action = min_action
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.update_period = update_period
        self.tau = tau
        self.noise_std = noise_std
        self.warmup = warmup
        self.noise_clip = noise_clip
        self.name = name
        self.ckpt_dir = ckpt_dir

        model_name = f'{name}__' \
                     f'gamma_{gamma}__' \
                     f'alpha_{alpha}__' \
                     f'beta_{beta}__' \
                     f'fc1_{fc1_dim}__' \
                     f'fc2_{fc2_dim}__' \
                     f'bs_{batch_size}__' \
                     f'buffer_{memory_size}__' \
                     f'update_period_{update_period}__' \
                     f'tau_{tau}__' \
                     f'noise_std_{noise_std}__' \
                     f'warmup_{warmup}__' \
                     f'noise_clip_{noise_clip}__' \

        self.model_name = model_name
        self.learn_iter = 0
        self.full_path = os.path.join(self.ckpt_dir, self.model_name)

        # Initialize Replay Buffer
        self.replay_buffer = ReplayBuffer(self.memory_size, self.state_dims, self.action_dims)

        # Initialize Critics
        self.critic_1 = Critic(self.beta, self.state_dims, self.action_dims, self.fc1_dim, self.fc2_dim,
                               name='Critic_1', ckpt_dir=self.ckpt_dir)
        self.critic_2 = Critic(self.beta, self.state_dims, self.action_dims, self.fc1_dim, self.fc2_dim,
                               name='Critic_2', ckpt_dir=self.ckpt_dir)
        self.target_critic_1 = Critic(self.beta, self.state_dims, self.action_dims, self.fc1_dim,
                                      self.fc2_dim, name='Target_Critic_1', ckpt_dir=self.ckpt_dir)
        self.target_critic_2 = Critic(self.beta, self.state_dims, self.action_dims, self.fc1_dim,
                                      self.fc2_dim, name='Target_Critic_2', ckpt_dir=self.ckpt_dir)

        # Initialize Actor
        self.actor = Actor(self.alpha, self.state_dims, self.action_dims, self.fc1_dim, self.fc2_dim,
                           name='Actor', ckpt_dir=self.ckpt_dir)
        self.target_actor = Actor(self.alpha, self.state_dims, self.action_dims, self.fc1_dim, self.fc2_dim,
                                  name='Target_Actor', ckpt_dir=self.ckpt_dir)

        # Update network parameters
        self.update_parameters(tau=1)

    def choose_action(self, state, add_noise=True):
        if self.learn_iter < self.warmup:
            mu = T.tensor(np.random.rand(*self.action_dims) * (self.max_action[0] - self.min_action[0]) +
                          self.min_action[0], dtype=T.float).to(self.actor.device)
        else:
            state = T.tensor(state, dtype=T.float).to(self.actor.device)
            self.actor.eval()
            mu = self.actor.forward(state).to(self.actor.device)
            self.actor.train()

        if add_noise:
            noise = np.random.normal(0, self.noise_std, self.action_dims)
            noise = T.tensor(noise, dtype=T.float).to(self.actor.device)
            mu = T.clamp(T.add(mu, noise), self.min_action[0], self.max_action[0])
        return mu.cpu().detach().numpy()

    def store_transition(self, state, action, reward, state_, done):
        return self.replay_buffer.store_transition(state, action, reward, state_, done)

    def load_batch(self):
        states, actions, rewards, states_, done = self.replay_buffer.load_batch(self.batch_size)
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        done = T.tensor(done, dtype=T.bool).to(self.actor.device)
        return states, actions, rewards, states_, done

    def update_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # collect parameters from networks
        actor_params = self.actor.named_parameters()
        actor_target_params = self.target_actor.named_parameters()
        critic1_params = self.critic_1.named_parameters()
        critic1_target_params = self.target_critic_1.named_parameters()
        critic2_params = self.critic_2.named_parameters()
        critic2_target_params = self.target_critic_2.named_parameters()

        # convert parameters into state dictionaries
        actor_state_dict = dict(actor_params)
        actor_target_state_dict = dict(actor_target_params)
        critic1_state_dict = dict(critic1_params)
        critic1_target_state_dict = dict(critic1_target_params)
        critic2_state_dict = dict(critic2_params)
        critic2_target_state_dict = dict(critic2_target_params)

        # iterate and update
        for item in actor_state_dict:
            actor_state_dict[item] = tau * actor_state_dict[item].clone() + \
                                     (1 - tau) * actor_target_state_dict[item].clone()
        for item in critic1_state_dict:
            critic1_state_dict[item] = tau * critic1_state_dict[item].clone() + \
                                       (1 - tau) * critic1_target_state_dict[item].clone()
        for item in critic2_state_dict:
            critic2_state_dict[item] = tau * critic2_state_dict[item].clone() + \
                                       (1 - tau) * critic2_target_state_dict[item].clone()

        # load the new parameters as state dicts into the networks
        self.target_actor.load_state_dict(actor_state_dict)
        self.target_critic1.load_state_dict(critic1_state_dict)
        self.target_critic2.load_state_dict(critic2_state_dict)

    def save_model(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_model(self, gpu_to_cpu=False):
        # loading all networks
        self.actor.load_checkpoint(gpu_to_cpu=gpu_to_cpu)
        self.critic_1.load_checkpoint(gpu_to_cpu=gpu_to_cpu)
        self.critic_2.load_checkpoint(gpu_to_cpu=gpu_to_cpu)
        self.target_critic_1.load_checkpoint(gpu_to_cpu=gpu_to_cpu)
        self.target_critic_2.load_checkpoint(gpu_to_cpu=gpu_to_cpu)

    def learn(self):
        # learns only after warmup iteration and when there is at least a single batch in buffer to load
        if self.learn_iter < self.warmup or self.learn_iter < self.batch_size:
            return

        states, actions, rewards, states_, done = self.load_batch()

        # compute the noisy next actions
        action_noise = np.random.normal(0, scale=self.noise_std, size=(self.batch_size, *self.action_dims))
        action_noise = T.clamp(T.tensor(action_noise, dtype=T.float), min=-self.noise_clip,
                               max=self.noise_clip).to(self.critic_1.device)
        actions_ = self.actor(states_).to(self.critic_1.device)
        actions_ = T.clamp(T.add(actions_, action_noise), min=self.min_action[0], max=self.max_action[0])

        # compute the state action values
        q1 = self.critic_1.forward(states, actions)
        q2 = self.critic_2.forward(states, actions)
        q1_ = self.critic_1.forward(states_, actions_)
        q2_ = self.critic_2.forward(states_, actions_)
        q1_[done] = 0.0
        q2_[done] = 0.0
        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)
        q_min = T.min(q1_, q2_).to(self.critic_1.device)
        y = rewards + self.gamma * q_min
        y = y.view(self.batch_size, 1)

        # optimize the critic networks
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_1_loss = F.mse_loss(y, q1)
        critic_2_loss = F.mse_loss(y, q2)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # check if time to actor learn and networks update
        self.learn_iter += 1
        if self.learn_iter % self.update_period != 0:
            return

        # optimize actor network
        new_actions = self.actor.forward(states)
        self.actor.optimizer.zero_grad()
        actor_loss = -T.mean(self.critic_1.forward(states, new_actions))
        actor_loss.backward()
        self.actor.optimizer.step()

        # update networks parameters
        self.update_parameters()


