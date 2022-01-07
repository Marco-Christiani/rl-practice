"""
Usage: python main.py train
"""
import datetime
import os.path
from typing import Dict

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
from torch.utils.tensorboard import SummaryWriter
from network import Network
from helpers import pre_proc, e_greedy

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


def compute_q_target(policy_net: torch.nn.Module,
                     batch_next_obs: torch.tensor,
                     batch_dones: torch.tensor,
                     rewards: torch.tensor, gamma: float):
    # get next values / boostrap estimate
    next_qs = policy_net(batch_next_obs)
    # get max Q of next step. IE: max_a(Q(s_t+1, a))
    max_next_qs = next_qs.max(dim=1).values
    # mask next q_qs with dones (if terminal state, no bootstrap estimate)
    td_target = rewards.flatten() + (
                1.0 - batch_dones.flatten()) * gamma * max_next_qs.flatten()
    return td_target


def update_policy(policy_net: nn.Module,
                  batch: Dict[str, np.ndarray],
                  criterion: nn.Module,
                  optimizer: optim,
                  gamma: float):
    optimizer.zero_grad()
    curr_qs = policy_net(torch.Tensor(batch['obs'])).gather(index=torch.tensor(batch['act'], dtype=torch.long), dim=1)
    with torch.no_grad():
        target_qs = compute_q_target(policy_net,
                                     torch.Tensor(batch['next_obs']),
                                     torch.Tensor(batch['done']),
                                     torch.Tensor(batch['rew']), gamma)
    loss = criterion(curr_qs.flatten(), target_qs)
    loss.backward()
    optimizer.step()
    return loss


@cli.command()
@click.option("--learning_rate", default=0.0001, show_default=True)
@click.option("--gamma", default=0.98, show_default=True)
@click.option("--buffer_size", default=200, show_default=True)
@click.option("--batch_size", default=200, show_default=True)
@click.option("--save_dir", default="../saves/", show_default=True)
@click.option("--tau", default=100, show_default=True)
@click.option("--reward_scale", default=100, show_default=True)
@click.option("--egreedy", is_flag=True, default=False, show_default=True)
@click.option("--load_path", default=None, show_default=True)
@click.option("--tag", default=None, show_default=True)
def train(
        learning_rate: float,
        gamma: float,
        buffer_size: int,
        batch_size: int,
        save_dir: str,
        tau: int,
        reward_scale: int,
        egreedy: bool = False,
        tag: str = None,
        load_path=None):
    tb_writer = SummaryWriter("/home/mchristiani/dqn_logs/")
    saved_times = {}
    start_time = datetime.datetime.now()
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
    else:
        policy_net = Network(4, 2)


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
        while True:
            action = None
            if egreedy:
                action = e_greedy()
            if action is None:
                with torch.no_grad():
                    # select action with policy net
                    output = policy_net(prev_obs)
                    # select max q value action (argmax)
                    action = torch.argmax(output).item()

            action_array.append(action)
            next_obs, reward, done, info = env.step(action)
            next_obs = pre_proc(next_obs)
            episode_length += reward
            reward /= reward_scale  # scale reward
            rb.add(obs=prev_obs, act=action, rew=reward, next_obs=next_obs,
                   done=done)
            prev_obs = next_obs
            i += 1
            if done:
                prev_obs = pre_proc(env.reset())
                loss = update_policy(policy_net,
                                     rb.sample(batch_size),
                                     criterion,
                                     optimizer,
                                     gamma)
                loss_array.append(loss.item())
                rb.clear()
                tb_writer.add_scalar("loss", loss, i)
                tb_writer.add_scalar("episode_length", episode_length, i)
                tb_writer.add_scalar("avg_action", np.mean(action_array), i)
                tb_writer.flush()
                episode_length_array.append(episode_length)
                episode_length = 0
                action_array = []
            # if i % tau == 0:
            #     mean_loss = np.mean(loss_array)
            #     if mean_loss >= 150 and len(saved_times) == 0:
            #         saved_times["150"] = datetime.datetime.now()
            #     if mean_loss == 200 and len(saved_times) == 1:
            #         saved_times["200"] = datetime.datetime.now()
            #
            #     print(f"avg_loss: {mean_loss:.6f}",
            #           f"avg_ep_len: {np.mean(episode_length_array):.6f}",
            #           f"n_episodes: {len(episode_length_array)}",
            #           "iter:", i,
            #           f"avg_action: {np.mean(action_array):.6f}")
            #     loss_array = []
            #     episode_length_array = []
    finally:
        tb_writer.close()
        end_time = datetime.datetime.now()
        avg_action = np.mean(action_array)
        avg_loss = np.mean(loss_array)
        divider = '-'*100
        print()
        print(divider)
        print(tag)
        print(f'Done! {avg_loss=} {avg_action=}')
        print(f"Start Time: {start_time} | End Time: {end_time} | Elapsed: {end_time-start_time}")
        for ep_len, time in saved_times.items():
            print(f'Hit {ep_len} ep len at: {time}')
        fpath = os.path.join(save_dir, f"dqn-{tag}-{start_time}-{end_time}.pth")
        torch.save(policy_net, fpath)
        print('Saved to', fpath)
        print(divider)

        # import pandas as pd
        # df = pd.DataFrame(data=action_array, columns=['action'])
        # avgs = df.loc[:, 'action'].rolling(window=100).mean()
        # plt.plot(list(range(len(action_array))), action_array, '.b')
        # plt.plot(list(range(len(avgs))), avgs, '-r')
        # plt.show()


@cli.command()
@click.option("--n_games", default=1, show_default=True)
@click.option("--load_path", default="../out.pth", show_default=True)
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
