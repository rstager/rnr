import numpy as np
import random
from rl.memory import SequentialMemory

from rnr.segment_tree import MinSegmentTree, SumSegmentTree


class PrioritizedMemory(SequentialMemory):
    def __init__(self, limit, priority, alpha=1.0, callback=None,**kwargs):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        limit: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedMemory, self).__init__(limit,**kwargs)
        assert alpha > 0
        self._alpha = alpha
        self.priority=priority
        self.callback=callback

        it_capacity = 1
        while it_capacity < limit:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def append(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx=super(PrioritizedMemory, self).append(*args,**kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, self.nb_entries - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, batch_idxs=None, beta=None):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
            return includes weights and idxs if sp

        Returns
        -------
        experiences:
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """

        if beta: assert beta > 0

        if not batch_idxs: batch_idxs = self._sample_proportional(batch_size)


        experiences=super(PrioritizedMemory, self).sample(batch_size,batch_idxs)

        if self.priority:
            self.update_priorities(batch_idxs, self.priority.priority(experiences))
        if not beta:
            if self.callback: self.callback(experiences)
            return experiences

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.nb_entries) ** (-beta)

        for idx in batch_idxs:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.nb_entries) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        return tuple(experiences + [weights, batch_idxs])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.nb_entries
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config['alpha'] = self.alpha
        return config


class SigmaPriority():
    def __init__(self,agent=None):
        self.agent=agent
    def priority(self, experiences):
        assert self.agent != None
        state0=np.empty([len(experiences),1,*experiences[0].state0[0].shape])
        state1=np.empty([len(experiences),1,*experiences[0].state1[0].shape])
        action=np.empty([len(experiences),*experiences[0].action.shape])
        reward=np.empty([len(experiences)])
        terminal1=np.empty([len(experiences)])
        for idx,e in enumerate(experiences):
            state0[idx]=e.state0
            state1[idx]=e.state1
            reward[idx]=e.reward
            action[idx]=e.action
            terminal1[idx]=e.terminal1
        a=self.agent.target_actor.predict(state1)
        q0=self.agent.critic.predict([action,state0])
        q1=self.agent.target_critic.predict([ a, state1])
        sigma= q0[:,0] - (reward  + self.agent.gamma * q1[:,0])
        # how should we normalize sigma and avoid outliers?
        ret=np.minimum(np.abs(sigma),1.0)
        #print("sigma {}".format(ret))
        return ret
