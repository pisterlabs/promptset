# pylint: disable=no-member
import numpy as np 
import torch 
import random 
import bisect
def combined_shape(length, shape=None):
    # taken from openAI spinning up. 
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class SortedBuffer:
    """
    Buffer that efficiently remains sorted.  
    """
    def __init__(self, obs_dim, act_dim, size,
                use_td_lambda_buf=False ):
        self.obs_buf = None
        self.obs2_buf= None
        self.act_buf= None
        self.discounted_rew_to_go_buf= None
        self.cum_rew= None
        self.horizon= None
        self.rollout_length = None
        self.buffer_dict = dict(obs=self.obs_buf, obs2=self.obs2_buf, 
                                act=self.act_buf, 
                                discounted_rew_to_go=self.discounted_rew_to_go_buf,
                                cum_rew=self.cum_rew, horizon=self.horizon, 
                                rollout_length=self.rollout_length
                                )
        
        self.use_td_lambda_buf = use_td_lambda_buf
        if self.use_td_lambda_buf:
            self.rollout_end_ind_buf = None
            self.raw_rew_buf = None
            
            self.buffer_dict['rollout_end_ind'] = self.rollout_end_ind_buf
            self.buffer_dict['raw_rew'] = self.raw_rew_buf
        
        self.size, self.max_size = 0, size
        self.total_num_steps_added = 0

    def add_rollouts(self, list_of_rollout_dicts):
        # sorted in ascending order. largest values at the back. 
        for rollout in list_of_rollout_dicts:
            # dont bother adding rollouts that are worse than the worst in the buffer. 
            # but still count them to the overall number of rollouts seen. 
            len_rollout = len(rollout['terminal'])
            self.total_num_steps_added += len_rollout
            if self.size == self.max_size and rollout['cum_rew'][0] <= self.buffer_dict['cum_rew'][-1]:
                continue 
            
            self.size = min(self.size+len_rollout, self.max_size)
            
            if self.buffer_dict['obs'] is not None:
                # find where everything from this rollout should be inserted into 
                # each of the numpy buffers. Uses the cumulative/terminal rewards
                # minus so that highest values are at the front. 
                sort_ind = np.searchsorted(-self.buffer_dict['cum_rew'], -rollout['cum_rew'][0]  )
                end_ind = len_rollout+sort_ind 
            else: 
                end_ind = len_rollout 
            
            if self.use_td_lambda_buf:
                # will be appended and treated like everything else. 
                end_ind = np.repeat(end_ind, len_rollout)
                rollout['rollout_end_ind'] = end_ind

            for key in self.buffer_dict.keys(): 
                # NOTE: assumes that buffer and rollout use the same keys!
                # needed at init!
                if self.buffer_dict[key] is None:
                    self.buffer_dict[key] = rollout[key]
                else: 
                    self.buffer_dict[key] = np.insert(self.buffer_dict[key], sort_ind, rollout[key], axis=0)

                if key == 'rollout_end_ind': 
                    self.buffer_dict[key][end_ind[0]:] = self.buffer_dict[key][end_ind[0]:]+len_rollout

                if self.size >= self.max_size:
                    # buffer is full. Need to trim!
                    # this will have a bias in that it will favour 
                    # the longer horizons at the end of the training data
                    # but it shouldnt make a major diff. 
                    self.buffer_dict[key] = self.buffer_dict[key][:self.max_size]

    def retrieve_path(self, start_index):
        end_index = self.buffer_dict['rollout_end_ind'][start_index]
        if end_index<= start_index:
            print("for sorted buffer shouldnt have looping here!")
            # we have looping 
            obs = np.concatenate( [self.buffer_dict['obs'][start_index:], self.buffer_dict['obs'][:end_index]], axis=0)
            rew = np.concatenate( [self.buffer_dict['raw_rew'][start_index:], self.buffer_dict['raw_rew'][:end_index]], axis=0)
        else: 
            obs = self.buffer_dict['obs'][start_index:end_index]
            rew = self.buffer_dict['raw_rew'][start_index:end_index]

        return torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(rew, dtype=torch.float32)


    def get_desires(self, last_few = 75):
        """
        This function calculates the new desired reward and new desired horizon based on the replay buffer.
        New desired horizon is calculted by the mean length of the best last X episodes. 
        New desired reward is sampled from a uniform distribution given the mean and the std calculated from the last best X performances.
        where X is the hyperparameter last_few.
        """
        # it is the first occurence of each cumulative ind. 
        unique_cum_rews, unique_cum_inds = np.unique(-self.buffer_dict['cum_rew'], return_index=True)
        #unique returns a sorted dictionary need to reverse it. 
        unique_cum_rews, unique_cum_inds = -unique_cum_rews[:last_few], unique_cum_inds[:last_few]
        #The exploratory desired horizon dh0 is set to the mean of the lengths of the selected episodes
        new_desired_horizon = round( self.buffer_dict['rollout_length'][unique_cum_inds].mean()  )

        # from these returns calc the mean and std
        mean_returns = np.mean(unique_cum_rews)
        std_returns = np.std(unique_cum_rews) 

        return mean_returns, std_returns, new_desired_horizon

    def __getitem__(self, idx):
        # turn this into a random value!
        #rand_ind = np.random.randint(0,self.size) # up to current max size. 
        return self.sample_batch(idxs=idx)

    def __len__(self):
        return self.size #self.num_batches_per_epoch

    def sample_batch(self, idxs=None, batch_size=256):
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)
        return {key:torch.as_tensor(arr[idxs],dtype=torch.float32) for key, arr in self.buffer_dict.items()}


class RingBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    # Taken from OpenAI spinning up. 
    """
    def __init__(self, obs_dim, act_dim, size, use_td_lambda_buf=False):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        if act_dim==1:
            self.act_buf = np.zeros(size, dtype=np.float32)
        else: 
            self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.discounted_rew_to_go_buf = np.zeros(size, dtype=np.float32)
        self.cum_rew = np.zeros(size, dtype=np.float32)
        self.horizon_buf = np.zeros(size, dtype=np.float32)

        self.buf_list = [self.obs_buf, 
                        self.obs2_buf, 
                        self.discounted_rew_to_go_buf,
                        self.act_buf, 
                        self.cum_rew, 
                        self.horizon_buf,
                        ]

        self.value_names = ['obs', 
                            'obs2', 
                            'discounted_rew_to_go', 
                            'act', 
                            'cum_rew', 
                            'horizon',
                            ]

        self.use_td_lambda_buf = use_td_lambda_buf
        if self.use_td_lambda_buf:
            self.rollout_end_ind_buf = np.zeros(size, dtype=np.int32)
            self.raw_rew_buf = np.zeros(size, dtype=np.float32)
            
            self.buf_list.append(self.raw_rew_buf)
            self.value_names.append('raw_rew')

        self.ptr, self.size, self.max_size = 0, 0, size
        self.total_num_steps_added = 0

    def retrieve_path(self, start_index):
        end_index = self.rollout_end_ind_buf[start_index]
        if end_index<= start_index:
            # we have looping
            obs = np.concatenate( [self.obs_buf[start_index:], self.obs_buf[:end_index]], axis=0)
            rew = np.concatenate( [self.raw_rew_buf[start_index:], self.raw_rew_buf[:end_index]], axis=0)
        else: 
            obs = self.obs_buf[start_index:end_index]
            rew = self.raw_rew_buf[start_index:end_index]

        return torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(rew, dtype=torch.float32)

    def add_to_buffer(self,np_buf, iters_adding, data ):
        if (self.ptr+iters_adding)>self.max_size:
            amount_pre_loop = self.max_size-self.ptr
            amount_post_loop = iters_adding-amount_pre_loop
            np_buf[self.ptr:] = data[:amount_pre_loop]
            np_buf[:amount_post_loop] = data[amount_pre_loop:]
        else: 
            np_buf[self.ptr:self.ptr+iters_adding] = data

    def add_rollouts(self, list_of_rollout_dicts):
        for rollout in list_of_rollout_dicts:
            iters_adding = len(rollout['terminal'])
            self.total_num_steps_added += iters_adding

            for np_buf, key in zip(self.buf_list,
                                self.value_names ):
                self.add_to_buffer(np_buf, iters_adding, rollout[key])
                
            if self.use_td_lambda_buf:
                end_ind = int((self.ptr+iters_adding) % self.max_size)
                assert end_ind >=0
                end_ind = np.repeat(end_ind, iters_adding)
                self.add_to_buffer(self.rollout_end_ind_buf, iters_adding, end_ind)

            self.ptr = (self.ptr+iters_adding) % self.max_size
            self.size = min(self.size+iters_adding, self.max_size)

    def __getitem__(self, idx):
        return self.sample_batch(idxs=idx)

    def __len__(self):
        return self.size

    def sample_batch(self, idxs=None, batch_size=32):
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     discounted_rew_to_go=self.discounted_rew_to_go_buf[idxs],
                     horizon=self.horizon_buf[idxs],
                     cum_rew=self.cum_rew[idxs],
                     )
        if self.use_td_lambda_buf:
            batch['start_index'] = idxs
            batch['rollout_end_ind'] = self.rollout_end_ind_buf[idxs]
            batch['raw_rew'] = self.raw_rew_buf[idxs]
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}