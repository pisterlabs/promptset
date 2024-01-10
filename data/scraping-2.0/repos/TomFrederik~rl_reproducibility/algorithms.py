"""Note: many functions here take a "memory" argument.
See the documentation of sample_memory in utils.py for
an explanation"""
import numpy as np
import copy
import torch
import utils


class TargetAlgorithm:
    """An algorithm for providing targets for the NPG or TRPO updates.

    Child classes need to overwrite targets() and should overwrite train()
    if they require training.

    Examples: MC returns, Critic baseline, GAE"""

    def __init__(self, gamma=1):
        self.gamma = gamma

    def train(self, memory):
        """Train the critic based on some sampled memory.

        This should be called during training procedures but
        child classes don't have to implement it if they don't require
        training (for example MC returns)."""
        pass

    def targets(self, memory):
        """Return value targets for the state-action pairs from memory.

        Returns:
            A torch tensor of shape (N, ) where N is the number of steps in the memory
            The n-th element should be an estimate of the return from the n-th memory
            step until the end of its episode (optionally with a baseline subtracted).
            """
        raise NotImplementedError


class ActorOnlyMC(TargetAlgorithm):
    """No critic, targets are simply sampled returns (REINFORCE targets)"""

    def targets(self, memory):
        rewards = memory[2]
        masks = memory[3]

        return get_returns(rewards, masks, self.gamma)


class GAE(TargetAlgorithm):
    '''
    Computes GAE as the target
    '''

    def __init__(self, critic, critic_optim, gamma=1, lamda=1, batch_size=16, epochs=5):
        '''
        Args:
            critic: a Critic instance. critic.forward(states) should return state value estimates
            critic_optim: an optimizer instance from torch.optim, optimizing critic's parameters
            gamma: discount_factor
            lamda: weighting factor for the n-step advantages
            batch_size: batch_size to use for training the critic
            epochs: number of epochs to train on a fixed memory
        '''
        super().__init__(gamma)
        self.critic = critic
        self.optim = critic_optim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lamda = lamda
    
    def targets(self, memory):
        self.critic.eval()
        (states, _, rewards, masks) = memory

        targets = get_gae(states, rewards, masks, self.gamma, self.lamda, self.critic)

        return targets

    def train(self, memory):
        self.critic.train()
        (states, _, rewards, masks) = memory

        returns = get_returns(rewards, masks, self.gamma)

        criterion = torch.nn.MSELoss()
        n = len(states)
        arr = np.arange(n)

        for epoch in range(self.epochs):
            np.random.shuffle(arr)

            for i in range(n // self.batch_size):
                batch_index = arr[self.batch_size * i: self.batch_size * (i + 1)]
                batch_index = torch.LongTensor(batch_index)
                inputs = states.float()[batch_index]
                target = returns.unsqueeze(1)[batch_index]

                values = self.critic(inputs)
                loss = criterion(values, target).float()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

class BaselineCriticMC(TargetAlgorithm):
    """Baseline state value function learned by a critic is subtracted froM REINFORCE targets."""

    def __init__(self, critic, critic_optim, gamma=1, batch_size=16, epochs=5):
        """Args:
            critic: a Critic instance. critic.forward(states) should return state value estimates
            critic_optim: an optimizer instance from torch.optim, optimizing critic's parameters
            gamma: discount_factor
            batch_size: batch_size to use for training the critic
            epochs: number of epochs to train on a fixed memory"""

        super().__init__(gamma)
        self.critic = critic
        self.optim = critic_optim
        self.batch_size = batch_size
        self.epochs = epochs

    def targets(self, memory):
        self.critic.eval()
        states = memory[0]
        rewards = memory[2]
        masks = memory[3]

        return get_returns(rewards, masks, self.gamma) - self.critic(states.float())

    def train(self, memory):
        self.critic.train()
        rewards = memory[2]
        masks = memory[3]
        states = memory[0]

        returns = get_returns(rewards, masks, self.gamma)

        criterion = torch.nn.MSELoss()
        n = len(states)
        arr = np.arange(n)

        for epoch in range(self.epochs):
            np.random.shuffle(arr)

            for i in range(n // self.batch_size):
                batch_index = arr[self.batch_size * i: self.batch_size * (i + 1)]
                batch_index = torch.LongTensor(batch_index)
                inputs = states.float()[batch_index]
                target = returns.unsqueeze(1)[batch_index]

                values = self.critic(inputs)
                loss = criterion(values, target)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()


class ActorAlgorithm:
    """An algorithm for training an actor, currently either NPG or TRPO.

    Usage: Initialize the class with an Actor and TargetAlgorithm instance.
    Sample some memory and train the actor on it. Repeat several times.
    This class is only for training the actor, no other methods for interacting
    with it are provided.

    Child classes must overwrite step() which decides on the step to take
    based on the natural gradient and other parameters."""

    def __init__(self, actor, target_alg, *args):
        """Args:
            actor: an Actor instance
            target_alg: a TargetAlgorithm instance to use for determining update targets during training"""
        self.target_alg = target_alg
        self.actor = actor

    def train(self, memory):
        """Train the actor based on sampled memory.
        Modifies the actor that was provided in __init__.

        Uses the TargetAlgorithm specified in __init__ to determine update targets,
        then calculates loss and natural gradient. Step size then depends on subclass (NPG or TRPO)."""
        self.actor.train()
        states = memory[0]
        actions = memory[1]

        # ----------------------------
        # step 1: get targets
        returns = self.target_alg.targets(memory)

        # ----------------------------
        # step 3: get gradient of loss and hessian of kl
        loss = self.get_loss(returns, states, actions)
        loss_grad = torch.autograd.grad(loss, self.actor.parameters())
        loss_grad = utils.flat_grad(loss_grad)
        step_dir = self.conjugate_gradient(states, loss_grad.data, nsteps=10)

        # ----------------------------
        # step 4: get step direction and step size and update actor
        results = self.step(step_dir, states, actions, returns, loss, loss_grad)

        return results

    # Remaining functions aren't needed externally, only used inside this class
    def step(self, step_dir, states, *args):
        """Take a step in direction step_dir, based on states and possibly other arguments.

        Returns:
            (float) The step size, as a coefficient of step_dir"""
        raise NotImplementedError

    def get_loss(self, targets, states, actions, *args, **kwargs):
        log_policy = self.actor.get_log_probs(states.float(), actions)
        losses = targets * log_policy
        return losses.mean()

    def fisher_vector_product(self, states, p):
        p.detach()
        kl = self.actor.get_kl(states.float())
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad = utils.flat_grad(kl_grad)  # check kl_grad == 0

        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = torch.autograd.grad(kl_grad_p, self.actor.parameters())
        kl_hessian_p = utils.flat_hessian(kl_hessian_p)

        return kl_hessian_p + 0.1 * p

    # from openai baseline code
    # https://github.com/openai/baselines/blob/master/baselines/common/cg.py
    def conjugate_gradient(self, states, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = self.fisher_vector_product(states, p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x


class NPG(ActorAlgorithm):
    def __init__(self, actor, target_alg, lr):
        super().__init__(actor, target_alg)
        self.lr = lr

    def step(self, step_dir, states, *args):
        old_actor = copy.deepcopy(self.actor)
        
        params = utils.flat_params(self.actor)
        new_params = params + self.lr * step_dir
        utils.update_model(self.actor, new_params)

        kl = self.actor.get_kl(states.float(), old_actor=old_actor)
        kl = kl.mean()

        return {'kl':kl, 'step_size':self.lr, 'entropy':self.actor.get_entropy(states.float())} # return statistics


class TRPO(ActorAlgorithm):
    def __init__(self, actor, target_alg, max_kl):
        super().__init__(actor, target_alg)
        self.max_kl = max_kl

    def get_loss(self, targets, states, actions, old_policy=None):
        new_policy = self.actor.get_log_probs(states.float(), actions)
        if old_policy is None:
            old_policy = new_policy.detach().clone()
        else:
            old_policy = old_policy.detach()
        losses = targets * torch.exp(new_policy - old_policy)
        return losses.mean()

    def step(self, step_dir, states, actions, returns, loss, loss_grad):
        '''
        Returns the step size taken
        '''
        params = utils.flat_params(self.actor)
        shs = 0.5 * (step_dir * self.fisher_vector_product(states, step_dir)
                     ).sum(0, keepdim=True)
        step_size = 1 / torch.sqrt(shs / self.max_kl)[0]
        full_step = step_size * step_dir

        # ----------------------------
        # step 5: do backtracking line search for n times
        # Create a copy of the current actor
        old_actor = copy.deepcopy(self.actor)
        expected_improve = (loss_grad * full_step).sum(0, keepdim=True)
        old_policy = old_actor.get_log_probs(states.float(), actions)

        flag = False
        fraction = 1.0
        for i in range(10):
            new_params = params + fraction * full_step
            utils.update_model(self.actor, new_params)
            new_loss = self.get_loss(returns, states, actions, old_policy)
            loss_improve = new_loss - loss
            expected_improve *= 0.5
            kl = self.actor.get_kl(states.float(), old_actor=old_actor)
            kl = kl.mean()

            #print('kl: {:.4f}  loss improve: {:.4f}  expected improve: {:.4f}  '
            #   'number of line search: {}'
            #   .format(kl.data.numpy(), loss_improve, expected_improve[0], i))

            # see https: // en.wikipedia.org / wiki / Backtracking_line_search
            if kl < self.max_kl and (loss_improve / expected_improve) > 0.5:
                flag = True
                return {'kl':kl, 'step_size':fraction*step_size.item(), 'entropy':self.actor.get_entropy(states.float())} # return statistics
                

            fraction *= 0.5

        if not flag:
            params = utils.flat_params(old_actor)
            utils.update_model(self.actor, params)
            print('policy update does not impove the surrogate')

            return {'kl': 0, 'step_size': 0, 'entropy':self.actor.get_entropy(states.float())} # return statistics


def get_returns(rewards, masks, gamma=1):
    """Calculate the returns for each time step from sample memory.

    Args:
        rewards: Tensor of shape (N, )
        masks: Tensor of shape (N, ) with values 0/1 or False/True
        gamma: discount factor

    Returns:
        A Tensor of shape (N, ) of returns until the end of the episode."""
    returns = torch.zeros_like(rewards)

    running_returns = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + gamma * running_returns * masks[t]
        returns[t] = running_returns

    # Original implementation normalizes returns, do we want that?
    #returns = (returns - returns.mean()) / returns.std()
    return returns


def get_episode_returns(rewards, masks, gamma=1):
    """Calculate the returns for complete episodes from memory.

    Args:
        rewards: Tensor of shape (N, )
        masks: Tensor of shape (N, ) with values 0/1 or False/True
        gamma: discount factor

    Returns:
        An ndarray of shape (n_episodes, ) of returns"""
    returns = []

    running_returns = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + gamma * running_returns * masks[t]
        if t == 0 or not masks[t - 1]:
            # we are at the beginning of an episode
            returns.append(running_returns)

    return np.array(returns)


def get_gae(states, rewards, masks, gamma, lamda, critic):
    '''
    compute GAE, given a critic and memory of states, rewards, masks
    '''

    # estimate values for all states
    value_estimates = critic(states.float())
    
    A = torch.zeros(len(rewards))

    for t in reversed(range(0, len(rewards))):
        if not masks[t]:
            # we are at the end of an episode
            delta_t = rewards[t] - value_estimates[t]
            A[t] = delta_t
        else:
            delta_t = rewards[t] + gamma * value_estimates[t+1] - value_estimates[t]
            A[t] = delta_t + gamma * lamda * A[t+1]
    
    return A



