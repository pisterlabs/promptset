import numpy as np
from model import Actor
from utils.utils import *

def train_model(actor, memory, state_size, action_size, args):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])

    # ----------------------------
    # step 1: get returns
    returns = get_returns(rewards, masks, args.gamma)

    # ----------------------------
    # step 2: get gradient of loss and hessian of kl and step direction
    mu, std = actor(torch.Tensor(states))
    old_policy = log_prob_density(torch.Tensor(actions), mu, std)
    loss = surrogate_loss(actor, returns, states, old_policy.detach(), actions)
    
    loss_grad = torch.autograd.grad(loss, actor.parameters())
    loss_grad = flat_grad(loss_grad)
    loss = loss.data.numpy()
    
    step_dir = conjugate_gradient(actor, states, loss_grad.data, nsteps=10)

    # ----------------------------
    # step 3: get step-size alpha and maximal step
    sHs = 0.5 * (step_dir * hessian_vector_product(actor, states, step_dir)
                 ).sum(0, keepdim=True)
    step_size = torch.sqrt(2 * args.max_kl / sHs)[0]
    maximal_step = step_size * step_dir

    # ----------------------------
    # step 4: perform backtracking line search for n iteration
    old_actor = Actor(state_size, action_size, args)
    params = flat_params(actor)
    update_model(old_actor, params)
    
    # 구했던 maximal step만큼 parameter space에서 움직였을 때 예상되는 performance 변화
    expected_improve = (loss_grad * maximal_step).sum(0, keepdim=True)
    expected_improve = expected_improve.data.numpy()

    # Backtracking line search
    # see cvx 464p https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
    # additionally, https://en.wikipedia.org/wiki/Backtracking_line_search
    flag = False
    alpha = 0.5
    beta = 0.5
    t = 1.0

    for i in range(10):
        new_params = params + t * maximal_step
        update_model(actor, new_params)
        
        new_loss = surrogate_loss(actor, returns, states, old_policy.detach(), actions)
        new_loss = new_loss.data.numpy()

        loss_improve = new_loss - loss
        expected_improve *= t
        improve_condition = loss_improve / expected_improve

        kl = kl_divergence(old_actor=old_actor, new_actor=actor, states=states)
        kl = kl.mean()

        print('kl: {:.4f} | loss_improve: {:.4f} | expected_improve: {:.4f} '
              '| improve_condition: {:.4f} | number of line search: {}'
              .format(kl.data.numpy(), loss_improve, expected_improve[0], improve_condition, i))

        # kl-divergence와 expected_new_loss_grad와 함께 trust region 안에 있는지 밖에 있는지를 판단
        # trust region 안에 있으면 loop 탈출
        # max_kl = 0.01
        if kl < args.max_kl and improve_condition > alpha:
            flag = True
            break

        # trust region 밖에 있으면 maximal_step을 반만큼 쪼개서 다시 실시
        t *= beta

    if not flag:
        params = flat_params(old_actor)
        update_model(actor, params)
        print('policy update does not impove the surrogate')

def get_returns(rewards, masks, gamma):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)

    running_returns = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + gamma * running_returns * masks[t]
        returns[t] = running_returns

    returns = (returns - returns.mean()) / returns.std()
    return returns

def surrogate_loss(actor, returns, states, old_policy, actions):
    mu, std = actor(torch.Tensor(states))
    new_policy = log_prob_density(torch.Tensor(actions), mu, std)
    returns = returns.unsqueeze(1)

    surrogate = torch.exp(new_policy - old_policy) * returns
    surrogate = surrogate.mean()
    return surrogate

# from openai baseline code
# https://github.com/openai/baselines/blob/master/baselines/common/cg.py
def conjugate_gradient(actor, states, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = hessian_vector_product(actor, states, p)
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