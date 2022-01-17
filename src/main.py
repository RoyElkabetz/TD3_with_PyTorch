import os
import gym
import argparse
import numpy as np
import pybullet_envs
import torch as T
from gym import wrappers

from utils import plot_learning_curve
from agents import Agent

# Seed everything
seed = 216
T.manual_seed(seed)
np.random.seed(seed)


if __name__ == '__main__':
    # ========================================= SET ARGUMENTS =========================================
    parser = argparse.ArgumentParser(description='This code is implementing the Deep Reinforcement Learning'
                                                 ' algorithm Twin Delayed Deep Deterministic Policy Gradient (TD3) '
                                                 'for low dimensional environment states as inputs.')
    parser.add_argument('-play', type=bool, default=False,
                        choices=[True, False],
                        help='Choosing the mode of the agent, False for learning or True for playing.')
    parser.add_argument('-gamma', type=float, default=0.99,
                        help='Discount factor for the update rule')
    parser.add_argument('-alpha', type=float, default=1e-3,
                        help='The Actor network learning rate')
    parser.add_argument('-beta', type=float, default=1e-3,
                        help='The Critic and Value networks learning rate')
    parser.add_argument('-fc1_dim', type=int, default=400,
                        help='The dimension of the first Linear layer across all networks')
    parser.add_argument('-fc2_dim', type=int, default=300,
                        help='The dimension of the second Linear layer across all networks')
    parser.add_argument('-memory_size', type=int, default=1000000,
                        help='The Replay Buffer memory size')
    parser.add_argument('-batch_size', type=int, default=100,
                        help='The batch size')
    parser.add_argument('-tau', type=float, default=0.005,
                        help='The parameters update constant -- 1 for hard update, 0 for no update')
    parser.add_argument('-update_period', type=int, default=2,
                        help='The period for updating the networks weights')
    parser.add_argument('-noise_std', type=float, default=0.1,
                        help='The action noise standard deviation')
    parser.add_argument('-warmup', type=int, default=100,
                        help='Number of transitions passes before start learning')
    parser.add_argument('-noise_clip', type=float, default=0.5,
                        help='Limiting value for noise clipping')
    parser.add_argument('-n_games', type=int, default=200,
                        help='Number of games / episodes')
    parser.add_argument('-env_name', type=str, default='InvertedPendulumBulletEnv-v0',
                        help='The environment name, PyBullet or Gym')
    parser.add_argument('-load_checkpoint', type=bool, default=False,
                        help='Load a model checkpoint')
    parser.add_argument('-gpu_to_cpu', type=bool, default=False,
                        help='In case you would like to load to a CPU a model that were trained on a GPU, '
                             'set to True, else False')
    parser.add_argument('-dir', type=str, default='../tmp',
                        help='Path for loading and saving models and plots')
    parser.add_argument('-monitor', type=bool, default=False,
                        help='If True, a video is being saved for each episode')
    args = parser.parse_args()

    # ========================================= MAIN SCRIPT =========================================
    env = gym.make(args.env_name)
    dir_path = os.path.join(args.dir, args.env_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    agent = Agent(gamma=args.gamma, alpha=args.alpha, beta=args.beta, state_dims=env.observation_space.shape,
                  action_dims=env.action_space.shape, max_action=env.action_space.high[0],
                  min_action=env.action_space.low[0], fc1_dim=args.fc1_dim, fc2_dim=args.fc2_dim,
                  memory_size=args.memory_size, batch_size=args.batch_size, tau=args.tau,
                  update_period=args.update_period, noise_std=args.noise_std, noise_clip=args.noise_clip,
                  warmup=args.warmup, name='TD3_'+args.env_name, ckpt_dir=dir_path)

    scores, avg_scores = [], []
    best_score = -np.inf

    if args.play:
        env.render(mode='human')

    if args.monitor:
        env = wrappers.Monitor(env, os.path.join(args.dir, args.env_name),
                               video_callable=lambda episode_id: True, force=True)
    if args.load_checkpoint:
        agent.load_model(gpu_to_cpu=args.gpu_to_cpu)

    for game in range(args.n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            if args.play:
                action = agent.choose_action(observation, add_noise=False)
            else:
                action = agent.choose_action(observation, add_noise=True)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            if not args.play:
                agent.learn()
            observation = observation_
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        print(f'| Game: {game:6.0f} | Score: {score:10.2f} | Best score: {best_score:10.2f} | '
              f'Avg score {avg_score:10.2f} |')

        if avg_score > best_score:
            best_score = avg_score
            if not args.play:
                agent.save_model()
    env.close()

    if not args.play:
        plot_learning_curve(scores, agent.full_path)
