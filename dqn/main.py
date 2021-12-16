"""
Usage: python main.py train
"""

import gym
import torch
import numpy as np
import torch.nn as nn
import time
import torch.nn.functional as F
import torch.optim as optim
from cpprb import ReplayBuffer
import matplotlib.pyplot as plt
import click

from dqn.network import Network
from dqn.helpers import pre_proc, e_greedy

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


def compute_q_target(target_net: torch.nn.Module,
                     batch_next_obs: torch.tensor,
                     batch_dones: torch.tensor,
                     rewards: torch.tensor, gamma: float):
    # get next values / boostrap estimate
    next_qs = target_net(batch_next_obs)
    # get max Q of next step. IE: max_a(Q(s_t+1, a))
    max_next_qs = next_qs.max(dim=1).values.flatten()
    # mask next q_qs with dones (if terminal state, no bootstrap estimate)
    td_target = rewards.flatten() + (
                1.0 - batch_dones.flatten()) * gamma * max_next_qs
    return td_target


def update_policy(policy_net, target_net, batch, criterion, optimizer, gamma):
    optimizer.zero_grad()
    curr_qs = policy_net(torch.tensor(batch['obs'])).max(dim=1).values
    target_qs = compute_q_target(target_net,
                                 torch.Tensor(batch['obs']),
                                 torch.Tensor(batch['done']),
                                 torch.Tensor(batch['rew']), gamma)
    loss = criterion(curr_qs, target_qs)
    loss.backward()
    optimizer.step()
    return loss


@cli.command()
@click.option("--learning_rate", default=0.0001, show_default=True)
@click.option("--gamma", default=0.98, show_default=True)
@click.option("--buffer_size", default=10000, show_default=True)
@click.option("--batch_size", default=128, show_default=True)
@click.option("--save_path", default="./out.pth", show_default=True)
@click.option("--tau", default=1000, show_default=True)
@click.option("--reward_scale", default=5, show_default=True)
@click.option("--egreedy", is_flag=True, default=False, show_default=True)
@click.option("--load_path", default=None, show_default=True)
def train(
        learning_rate,
        gamma,
        buffer_size,
        batch_size,
        save_path,
        tau,
        reward_scale,
        egreedy=False,
        load_path=None):
    env = gym.make("CartPole-v0")
    rb = ReplayBuffer(buffer_size,
                      env_dict={"obs": {"shape": 4},
                                "act": {"shape": 1},
                                "rew": {"shape": 1},
                                "next_obs": {"shape": 4},
                                "done": {"shape": 1}}
                      )
    if load_path:
        policy_net = torch.load(load_path)
        target_net = torch.load(load_path)
    else:
        policy_net = Network(4, 2)
        target_net = Network(4, 2)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # dont update this network on backprop

    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    optimizer = optim.SGD(policy_net.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(policy_net.parameters(), lr=learning_rate)
    prev_obs = pre_proc(env.reset())
    i = 1
    episode_length = 0
    episode_length_array = []
    loss_array = []
    action_array = []
    try:
        while episode_length < 200:
            action = None
            if egreedy:
                action = e_greedy(i)
            if not action:
                # select action with policy net
                output = policy_net(prev_obs)
                # select max q value action (argmax)
                action = output.max(dim=0).indices.item()
            action_array.append(action)
            next_obs, reward, done, info = env.step(action)
            episode_length += reward
            next_obs = pre_proc(next_obs)
            reward /= reward_scale  # scale reward
            rb.add(obs=prev_obs, act=action, rew=reward, next_obs=next_obs,
                   done=done)
            prev_obs = next_obs
            if done:
                prev_obs = pre_proc(env.reset())
                rb.on_episode_end()
                episode_length_array.append(episode_length)
                episode_length = 0
            i += 1
            if i >= buffer_size:
                loss = update_policy(policy_net,
                                     target_net,
                                     rb.sample(batch_size),
                                     criterion,
                                     optimizer,
                                     gamma)
                loss_array.append(loss.item())

                if i % tau == 0:
                    # update target net every tau iterations
                    target_net.load_state_dict(policy_net.state_dict())

                    print(f"avg_loss: {np.mean(loss_array):.6f}",
                          f"avg_ep_len: {np.mean(episode_length_array):.6f}",
                          f"n_episodes: {len(episode_length_array)}",
                          "iter:", i,
                          f"avg_action: {np.mean(action_array):.6f}")
                    loss_array = []
                    action_array = []
                    episode_length_array = []
    finally:
        avg_action = np.mean(action_array)
        avg_loss = np.mean(loss_array)
        print(f'Done! {avg_loss=} {avg_action=}')
        torch.save(target_net, save_path)
        print('Saved to', save_path)


@cli.command()
@click.option("--n_games", default=1, show_default=True)
@click.option("--load_path", default="./out.pth", show_default=True)
@click.option("--render", is_flag=False, show_default=True)
def test(load_path, n_games, render):
    env = gym.make("CartPole-v0")
    net = torch.load(load_path)
    net.eval()

    done = False
    rewards = []
    actions = []
    q_values = []
    with torch.no_grad():
        for _ in range(n_games):
            obs = env.reset()
            obs = pre_proc(obs)
            output = net(obs)
            done = 0
            while not done:
                action = output.max(dim=0).indices.item()
                obs, reward, done, info = env.step(action)
                if render:
                    env.render()
                    time.sleep(0.05)
                obs = pre_proc(obs)
                output = net(obs)
                rewards.append(reward)
                actions.append(action)
                q_values.append(list(output.numpy()))
    env.close()
    x = np.arange(len(actions))
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot([x[0] for x in q_values], '.')
    ax1.set_title('Action 1')
    ax1.set_ylabel('Q')
    ax2.plot([x[1] for x in q_values], '.')
    ax2.set_title('Action 2')
    ax2.set_ylabel('Q')
    plt.show()
    print(q_values)
    print(actions)


if __name__ == "__main__":
    cli()
