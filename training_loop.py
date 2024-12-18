"""
Training loop using DeepMind Control Suite to see how the replay buffer works
"""

import learning_utils.DataTrackers
import learning_utils.Memories
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from dm_control import suite
import numpy as np
import argparse
import cv2

from ReplayBuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Parse args
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='Name of this run', required=True)
    parser.add_argument('-e', '--episodes', help='Number of episodes to train for', required=True)
    parser.add_argument('-b', '--batch_size', help='Batch size to sample from the replay buffer', required=True)
    parser.add_argument('-r', '--report_freq', help='How often to save a report video', required=True)
    args = parser.parse_args()
    return args

## Utility function to visualize training
def make_video(frames, file_path):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_name = f'{file_path}.avi'
    fps = 30

    out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()

## Networks
class Actor(nn.Module):
    """
    Actor returrns mean and standard deviation for each action parameter
    """
    def __init__(self, observation_size, action_size):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        ).to(device)
        self.log_std = nn.Parameter(torch.zeros(action_size)).to(device)

    def forward(self, state):
        mean = self.net(state)
        std = self.log_std.exp()
        return mean, std

class Critic(nn.Module):
    """
    Critic returns the state-value function for the current state
    """
    def __init__(self, observation_size):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)

    def forward(self, state):
        return self.net(state)
    
## Agents
class AgentMPO:
    """
    Agent using MPO

    args:
        observations_size, action_size (int): Size of observation and action vectors
        epsilon (float): Controls the KL-divergence constraint
        gamma (float): Discount factor for past experiences
    """
    def __init__(self, observation_size, action_size, epsilon=0.1, gamma = 0.95, tau = 0.001):
        # Networks
        self.actor = Actor(observation_size, action_size)
        self.actor_target = Actor(observation_size, action_size)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(observation_size)
        self.critic_target = Critic(observation_size)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        # Parameters
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau

        # Lagrange multipliers for constraints
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True).to(device))
        self.log_beta = nn.Parameter(torch.zeros(1, requires_grad=True).to(device))
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.beta_optimizer = optim.Adam([self.log_beta], lr=3e-4)

    def forward(self, observation):
        """
        Returns an action by doing a forward pass on the actor network

        args:
            observation (tensor of size (self.observation_size, )): Unbatched single observation for a forward pass
        """
        # Batch the tensor
        observation = torch.FloatTensor(observation).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, std = self.actor(observation)
        action = torch.normal(mean, std)
        return action.cpu().squeeze().detach().numpy()
    
    def update(self, states, actions, rewards, next_states, dones):
        """
        Updates the actor based on a replay buffer sample
        
        args:
            states, actions, rewards, next_states, dones (tensor)

        returns:
            metrics (dictionary): critic_loss, actor_loss, kl_divergence, alpha, beta
        """
        # Compute critic loss
        with torch.no_grad():
            next_value_target = self.critic_target(next_states)
            q_targets = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * 0.99 * next_value_target

        q_values = self.critic(states)

        critic_loss = nn.functional.mse_loss(q_values, q_targets)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        # (1) Get the log probabilities of the actions given the current actor network
        mean, std = self.actor(states)
        current_actor_dist = distributions.Normal(mean, std)
        current_actor_logprob = current_actor_dist.log_prob(actions)

        # (2) Compute how important each replay buffer sample is for our update
        with torch.no_grad():
            tgt_mean, tgt_std = self.actor_target(states)
            target_actor_dist = distributions.Normal(tgt_mean, tgt_std)
            target_actor_logprop = target_actor_dist.log_prob(actions)
            weights = torch.exp(current_actor_logprob - target_actor_logprop)

        # (3) Compute KL divergence (distance between current actor and target actor distributions)
        kl_div = distributions.kl.kl_divergence(current_actor_dist, target_actor_dist).mean()

        # (4) Compute objectives with Lagrange multipliers
        alpha = torch.exp(self.log_alpha)
        beta = torch.exp(self.log_beta)

        with torch.no_grad():
            q_values = self.critic(states)

        actor_objective = torch.mean(weights * q_values)
        alpha_loss = -alpha * (kl_div - self.epsilon)
        beta_loss = -beta * (kl_div - self.epsilon)

        # (5) Combined actor loss
        actor_loss = -(actor_objective - alpha * kl_div - beta * (kl_div - self.epsilon))
        
        # (6) Update Lagrange multipliers
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.alpha_optimizer.step()
        
        self.beta_optimizer.zero_grad()
        beta_loss.backward(retain_graph=True)
        self.beta_optimizer.step()

        # (7) Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of the targets
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'kl_divergence': kl_div.item(),
            'alpha': alpha.item(),
            'beta': beta.item()
        }

    def _soft_update(self, network, target_network):
        """
        Soft update of target network
        
        Args:
            network (nn.Module): Source network
            target_network (nn.Module): Target network to be updated
        """
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        
## Fill the replay buffer 
def fill_replay_buffer(environment, agent, replay_buffer, batch_size):
    print("\nFilling replay buffer...")

    while replay_buffer.len() <= batch_size:
        # Reset episode
        time_step = env.reset()

        for t in range(500):
            # Get action and do it
            observation = time_step.observation['position']
            action = agent.forward(observation)
            time_step = env.step(action)

            # Store experience in memory
            reward = time_step.reward
            next_observation = time_step.observation['position']
            replay_buffer.add(observation, action, reward, next_observation, time_step.last())

## Training Loop
def train(environment, agent, replay_buffer, num_episodes, batch_size, name, report_freq):
    # Track the training loop metrics
    metrics = learning_utils.DataTrackers.TrainingLoopTracker("rewards", "actor_loss", "critic_loss")
    mpo_metrics = learning_utils.DataTrackers.TrainingLoopTracker("kl_divergence", "alpha", "beta")

    # Training loop
    for e in range(num_episodes):
        print("\nEpisode ", e)
        
        # Reset 
        time_step = environment.reset()
        total_reward = 0
        total_actor_loss = 0
        total_critic_loss = 0

        # Determine if we are reporting or not
        if e % report_freq == 0:
            reporting = True
            frames = []
        else:
            reporting = False

        for t in range(500):
            # Get action and do it
            observation = time_step.observation['position']
            action = agent.forward(observation)
            time_step = env.step(action)

            ## Store in replay buffer
            reward = time_step.reward
            next_observation = time_step.observation['position']
            replay_buffer.add(observation, action, reward, next_observation, time_step.last())

            # Render and capture frame 
            if reporting:
                # Adjust camera
                agent_pos = env.physics.named.data.xpos['torso']
                env.physics.named.data.cam_xpos['side'][0] = agent_pos[0]

                # Render and save frame
                frame = env.physics.render(camera_id = 'side')
                frames.append(frame)

            # Sample from replay buffer
            samp = replay_buffer.sample(batch_size)

            # Convert sample to stacked tensors
            sample = []
            for s in samp:
                s = [torch.tensor(a, dtype=torch.float32) for a in s]
                sample.append(torch.stack(s).to(device))

            states, actions, rewards, next_states, dones = sample

            # Update models
            update_metrics = agent.update(states, actions, rewards, next_states, dones)

            # Parse update_metrics
            critic_loss = update_metrics['critic_loss']
            actor_loss = update_metrics['actor_loss'] 
            kl_divergence = update_metrics['kl_divergence'] 
            alpha = update_metrics['alpha'] 
            beta = update_metrics['beta']

            # Update totals
            total_reward += reward
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss

            # Break if episode is over
            if time_step.last():
                break

        # Print and save metrics
        print('Total reward: ', total_reward)
        print('Total actor loss: ', total_actor_loss)
        print('Total critic loss: ', total_critic_loss)

        metrics.update(total_reward, total_actor_loss, total_critic_loss)
        metrics.save(f"results/{name}/metrics.csv")

        mpo_metrics.update(kl_divergence, alpha, beta)
        mpo_metrics.save(f"results/{name}/mpo_metrics.csv")

        # Make a progress report video once in a while
        if reporting:
            make_video(frames, f"results/{name}/ep{e}_progress_report")
        
    # Save final model
    torch.save(agent.actor.state_dict(), f'results/{name}/final_weights.pt')


if __name__ == "__main__":
    # Parse args
    args = argparser()
    batch_size = int(args.batch_size)
    episodes = int(args.episodes)
    report_freq = int(args.report_freq)

    # Load environment
    env = suite.load(domain_name="cheetah", task_name="run")

    # Load agent
    action_size = env.action_spec().shape[0]
    observation_size = env.observation_spec()['position'].shape[0]

    agent = AgentMPO(observation_size, action_size)

    # Load replay buffer 
    replay_buffer = learning_utils.Memories.ReplayBuffer(100000)
    fill_replay_buffer(env, agent, replay_buffer, batch_size)

    # Training loop
    train(env, agent, replay_buffer, episodes, batch_size, args.name, report_freq)

