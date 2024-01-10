import os
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.dataloader import DataLoader

# custom modules
from simplegpt.small.model import GPT
from simplegpt.small.config import GPTConfig


class Trainer:
    def __init__(
        self, 
        config:GPTConfig, 
        dataset:torch.utils.data.Dataset, 
        backend:str='nccl',
        out_dir:str='out',
        init_from:str='scratch',
        block_size:int=1024,
        weight_decay:float=1e-2, 
        learning_rate:float=2.5e-4, 
        betas:tuple=(0.9, 0.95), 
        compile_model:bool=False, 
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.backend = backend
        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay
        self.process_finished = False

        # validate config
        assert backend == 'nccl' or backend == 'gloo', 'backend must be either nccl or gloo' # only support these two backends for now
        assert init_from == 'scratch' or init_from == 'resume' or init_from.startswith('gpt-2'), 'init_from must be either scratch, resume, or a gpt-2 checkpoint'
        assert block_size <= config.block_size, 'block_size must be less than or equal to config.block_size'

        # is this a ddp run?
        ddp = int(os.environ.get('LOCAL_RANK', -1)) != -1

        if ddp:
            init_process_group(backend=backend)
            gpu_id = int(os.environ["LOCAL_RANK"])
            
            # determine the device we'll train on
            if config.device == 'auto':
                self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else 'cpu'
            else:
                self.device = config.device
        else:
            gpu_id = 0 # gpu_id 0 means this is the (single) master process, basically

            # determine the device we'll train on
            if config.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = config.device

        if gpu_id == 0:
            os.makedirs(out_dir, exist_ok=True)
        
        torch.manual_seed(1337 + gpu_id) # note: each worker gets a different seed
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

        # model init
        if init_from == 'scratch':
            # init a new model from scratch
            print("Initializing a new model from scratch")
            model = GPT(config)

        elif init_from == 'resume':
            print(f"Resuming training from {out_dir}")
            # resume training from a checkpoint.
            ckpt_path = os.path.join(out_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            model = GPT(config)
            model.load_state_dict(checkpoint['model'])
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']

        elif init_from.startswith('gpt2'):
            print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
            # initialize from OpenAI GPT-2 weights
            model = GPT.from_pretrained(init_from)

        # crop down the model block size if desired
        if block_size < model.block_size:
            model.crop_block_size(block_size)
        model = model.to(self.device)

        optimizer = model.configure_optimizers(weight_decay, learning_rate, betas)
        if init_from == 'resume':
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        # compile the model
        if compile_model:
            print("compiling the model... (takes a ~minute)")
            unoptimized_model = model
            model = torch.compile(model) # requires PyTorch 2.0
        
        # wrap model into DDP container
        if ddp:
            model = DDP(model, device_ids=[gpu_id])

        self.model = model
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0


    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)



    def run(self):
        if self.process_finished:
            print("Training process already finished, not running again")
            return

        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
        
        destroy_process_group()
        self.process_finished = True
