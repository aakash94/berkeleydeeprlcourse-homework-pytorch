import os
import argparse
import pickle
import numpy as np
import gym
import torch
import torch.nn as nn
import load_policy
from Agent import Agent
from Loader import Loader
from  tqdm import trange
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
from VisdomPlotter import VisdomPlotter

class DAgger:


    def __init__(self, env_string = "Hopper-v2", max_steps=-1, use_cuda = True, auto=True):
        # Note: Gotta set env before you set Agent
        self.env_string = env_string
        self.train_on_gpu = torch.cuda.is_available()
        if auto:
            self.auto_set_env(env_string=env_string)
            self.auto_set_agent()
            self.auto_set_expert(env_string=env_string)

        else:
            # This is not handled properly
            # you better things properly if auto is False
            pass
        self.max_steps = self.env.spec.timestep_limit
        if max_steps > 0:
            self.max_steps = max_steps

        self.data_by_expert = []
        self.data_by_agent = []

        if self.train_on_gpu:
            self.agent = self.agent.cuda()
        #self.agent = self.agent.double()
        self.agent = self.agent.float()


    def auto_set_env(self ,env_string = "Hopper-v2"):
        self.env = gym.make(env_string)


    def auto_set_agent(self):
        self.agent = Agent(input_size=np.prod(list(self.env.observation_space.shape)), output_size=np.prod(list(self.env.action_space.shape)))


    def auto_set_expert(self, env_string = "Hopper-v2"):
        policy_file =  "experts/"+env_string+".pkl"
        self.expert = load_policy.load_policy(policy_file)


    def save_agent(self):
        if self.train_on_gpu:
            self.agent = self.agent.cpu()

        self.agent.save_model(env_string=self.env_string)

        if self.train_on_gpu:
            self.agent = self.agent.cuda()


    def load_agent(self):
        if self.train_on_gpu:
            self.agent = self.agent.cpu()
        self.agent.load_model(env_string=self.env_string)
        if self.train_on_gpu:
            self.agent = self.agent.cuda()


    def demonstrate(self, num_episode = 1, expert_mode = False):
        # this is just to show off
        if not expert_mode:
            self.agent.eval()

        for i in range(num_episode):
            obs = self.env.reset()
            done = False
            steps = 0
            while not done:
                action = None
                ob = obs[None, :]
                if expert_mode:
                    action = self.expert(obs=ob)
                else:
                    ob = torch.from_numpy(ob).float()
                    if self.train_on_gpu:
                        ob = ob.cuda()
                    action = self.agent(ob)
                    if self.train_on_gpu:
                        action = action.cpu()
                    action = action.detach().numpy()
                obs, r, done, _ = self.env.step(action)
                steps +=1
                self.env.render()
                if steps >= self.max_steps:
                    break


    def get_demonstration_episode_count(self, num_episodes=1, expert_mode=False):

        total_reward = 0.0
        for i in range (num_episodes):
            obs = self.env.reset()
            done = False
            steps = 0
            while not done:
                ob = obs[None, :]
                action = None
                if expert_mode:
                    action = self.expert(ob)
                    self.data_by_expert.append((ob, action))
                else:
                    ob = torch.from_numpy(ob).float()
                    if self.train_on_gpu:
                        ob = ob.cuda()
                    action = self.agent(ob)
                    if self.train_on_gpu:
                        ob = ob.cpu()
                        action = action.cpu()
                    action = action.detach().numpy()
                    ob = ob.detach().double().numpy()
                    self.data_by_agent.append((ob))

                obs, r, done, _ = self.env.step(action)
                steps += 1
                total_reward += r
                if steps >= self.max_steps:
                    break

        average_reward = total_reward/num_episodes
        return average_reward


    def get_demonstration_step_count(self, num_steps=1, expert_mode=False):

        total_reward = 0.0
        total_steps = 0
        episodes = 0
        while total_steps < num_steps:
            obs = self.env.reset()
            done = False
            steps = 0
            episodes +=1
            while not done:
                ob = obs[None, :]
                action = None
                if expert_mode:
                    action = self.expert(ob)
                    self.data_by_expert.append((ob, action))
                else:
                    ob = torch.from_numpy(ob).float()
                    if self.train_on_gpu:
                        ob = ob.cuda()

                    action = self.agent(ob)

                    if self.train_on_gpu:
                        ob = ob.cpu()
                        action = action.cpu()
                    action = action.detach().numpy()
                    ob = ob.detach().double().numpy()
                    self.data_by_agent.append((ob))

                obs, r, done, _ = self.env.step(action)
                steps += 1
                total_steps += 1
                total_reward += r
                if steps >= self.max_steps:
                    break

        average_reward = total_reward / episodes
        return average_reward


    def get_demonstrations(self, num_episodes=-1, num_steps=-1, expert_mode=False ):
        # if both num_episodes and num_steps are gre                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    r                                                                                                                     s znvf                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ater than -1, num_episodes 1 will take effect.
        # collect data from expert experience
        if num_episodes == -1 and num_steps != -1 :
            return self.get_demonstration_step_count(num_steps=num_steps, expert_mode=expert_mode)
        elif num_episodes != -1 and num_steps == -1 :
            return self.get_demonstration_episode_count(num_episodes=num_episodes,expert_mode=expert_mode)
        else :
            print("Invalid parameters")
            return 0


    def teach_agent(self, epochs=1, learn_rate=0.0001, batch_size=128):

        expert_loader = Loader(data_collected=self.data_by_expert)
        optimizer = optim.Adam(self.agent.parameters(), lr=learn_rate)
        criterion = nn.MSELoss()
        loader = dataloader.DataLoader(expert_loader, batch_size=batch_size, shuffle=True)
        total_loss = 0.0
        for e in range(epochs):
            sum_loss = 0.0
            count = 0
            for x,y in loader:
                if self.train_on_gpu:
                    x = x.float().cuda()
                    y = y.float().cuda()
                optimizer.zero_grad()
                output = self.agent(x)
                loss = criterion(output, y)
                sum_loss += loss.item()
                count += 1
                loss.backward()
                optimizer.step()

            total_loss += (sum_loss/count)

        average_loss = total_loss/epochs
        return average_loss


    def expert_opinion(self):
        for ob in self.data_by_agent:
            action = self.expert(ob)
            self.data_by_expert.append((ob, action))


    def do_dagger(self, train_epochs = 1, collection_episodes = 1, total_loops = 1, expert_demo_count = 10):
        plotter = VisdomPlotter(env_name=self.env_string)
        self.get_demonstrations(num_episodes=expert_demo_count, expert_mode=True)
        for e in trange(total_loops):
            avg_loss = self.teach_agent(epochs=train_epochs)
            avg_reward = self.get_demonstrations(num_episodes=collection_episodes, expert_mode=False)

            self.expert_opinion()
            plotter.plot_line('average loss per epoch', 'loss', 'Training Loss', e, avg_loss)
            plotter.plot_line('average reward per run', 'reward', 'Reward by Agent', e, avg_reward)



if __name__ == '__main__':

    envs = ['Hopper-v2', 'Ant-v2', 'HalfCheetah-v2', 'Humanoid-v2', 'Reacher-v2', 'Walker2d-v2']
    environment = envs[0]

    da = DAgger(env_string=environment, auto=True)
    da.do_dagger(train_epochs=10, collection_episodes=10, total_loops=10, expert_demo_count=10)
    save_string = "da_" + environment
    da.agent.save_model(env_string=save_string)
    da.demonstrate(num_episode=10, expert_mode=False)

