import gym
import torch
import torch.nn as nn
import torch.nn.functional

from .rl_pg import TrainBatch, PolicyAgent, VanillaPolicyGradientLoss


class AACPolicyNet(nn.Module):
    def __init__(self, in_features: int, out_actions: int, **kw):
        """
        Create a model which represents the agent's policy.
        :param in_features: Number of input features (in one observation).
        :param out_actions: Number of output actions.
        :param kw: Any extra args needed to construct the model.
        """
        super().__init__()
        # TODO:
        #  Implement a dual-head neural net to approximate both the
        #  policy and value. You can have a common base part, or not.
        # ====== YOUR CODE: ======
        #qfunc/policy
        qlayers = []
        in_feat_num = in_features
        for out_feat_num in [64, 128, 256, 128, 64]:
            qlayers.append(nn.Linear(in_feat_num, out_feat_num))
            qlayers.append(nn.ReLU())
            in_feat_num = out_feat_num
        qlayers.append(nn.Linear(in_feat_num, out_actions))
        self.qnet = nn.Sequential(*qlayers)
        self.qnet.to(kw.get('device', 'cpu')) #check if we get gpu if not run it on cpu


        #vfunc
        vlayers = []
        in_feat_num = in_features
        for out_feat_num in [64, 128, 256, 128, 64]:
            vlayers.append(nn.Linear(in_feat_num, out_feat_num))
            vlayers.append(nn.ReLU())
            in_feat_num = out_feat_num
        vlayers.append(nn.Linear(in_feat_num, 1))
        self.vnet = nn.Sequential(*vlayers)
        self.vnet.to(kw.get('device', 'cpu')) #check if we get gpu if not run it on cpu
        # ========================

    def forward(self, x):
        """
        :param x: Batch of states, shape (N,O) where N is batch size and O
        is the observation dimension (features).
        :return: A tuple of action values (N,A) and state values (N,1) where
        A is is the number of possible actions.
        """
        # TODO:
        #  Implement the forward pass.
        #  calculate both the action scores (policy) and the value of the
        #  given state.
        # ====== YOUR CODE: ======
        action_scores = self.qnet(x)
        state_values = self.vnet(x)
        # ========================

        return action_scores, state_values

    @staticmethod
    def build_for_env(env: gym.Env, device="cpu", **kw):
        """
        Creates a A2cNet instance suitable for the given environment.
        :param env: The environment.
        :param kw: Extra hyperparameters.
        :return: An A2CPolicyNet instance.
        """
        # TODO: Implement according to docstring.
        # ====== YOUR CODE: ======
        net = AACPolicyNet(env.observation_space.shape[0], env.action_space.n, **kw)
        # ========================
        return net.to(device)


class AACPolicyAgent(PolicyAgent):
    def current_action_distribution(self) -> torch.Tensor:
        # TODO: Generate the distribution as described above.
        # ====== YOUR CODE: ======
        with torch.no_grad():
            curr = self.curr_state.view(1, -1)
            act_scores, _ = self.p_net.forward(curr)
           # print('#'*10)
            #print(net_out)
            probs = torch.softmax(act_scores, dim=-1)
            actions_proba = probs.view(-1)
        # ========================
        return actions_proba


class AACPolicyGradientLoss(VanillaPolicyGradientLoss):
    def __init__(self, delta: float):
        """
        Initializes an AAC loss function.
        :param delta: Scalar factor to apply to state-value loss.
        """
        super().__init__()
        self.delta = delta

    def forward(self, batch: TrainBatch, model_output, **kw):

        # Get both outputs of the AAC model
        action_scores, state_values = model_output

        # TODO: Calculate the policy loss loss_p, state-value loss loss_v and
        #  advantage vector per state.
        #  Use the helper functions in this class and its base.
        # ====== YOUR CODE: ======
        loss_v = self._value_loss(batch, state_values)
        advantage = self._policy_weight(batch, state_values)
        loss_p = self._policy_loss(batch, action_scores, advantage)
        # ========================

        loss_v *= self.delta
        loss_t = loss_p + loss_v
        return (
            loss_t,
            dict(
                loss_p=loss_p.item(),
                loss_v=loss_v.item(),
                adv_m=advantage.mean().item(),
            ),
        )

    def _policy_weight(self, batch: TrainBatch, state_values: torch.Tensor):
        # TODO:
        #  Calculate the weight term of the AAC policy gradient (advantage).
        #  Notice that we don't want to backprop errors from the policy
        #  loss into the state-value network.
        # ====== YOUR CODE: ======
        with torch.no_grad():
            advantage = batch.q_vals - state_values.detach()
        # ========================
        return advantage

    def _value_loss(self, batch: TrainBatch, state_values: torch.Tensor):
        # TODO: Calculate the state-value loss.
        # ====== YOUR CODE: ======
        val = state_values.view(-1)
        qval = batch.q_vals.view(-1)
        mse = nn.MSELoss(reduction='mean')
        loss_v = mse(qval, val)
        # ========================
        return loss_v
