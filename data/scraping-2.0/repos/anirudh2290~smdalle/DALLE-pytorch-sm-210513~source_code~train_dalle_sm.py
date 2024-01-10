import argparse
from pathlib import Path
import logging
import subprocess
import sys
import os
import json

import torch.multiprocessing as mp
import torch.distributed as dist

import torch
import wandb  # Quit early if user doesn't have wandb installed.
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE1024, DiscreteVAE, DALLE
from dalle_pytorch import distributed_utils
from dalle_pytorch.loader import TextImageDataset
from dalle_pytorch.tokenizer import tokenizer, HugTokenizer, ChineseTokenizer, YttmTokenizer, SimpleTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

from pathlib import Path
from random import randint, choice

import PIL

from torch.utils.data import Dataset
from torchvision import transforms as T

import psutil
import aws_util

from torchnet.dataset import SplitDataset
from torch.cuda.amp import autocast

try:
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel.torch.state_mod import state as smp_state
except ImportError:
    pass



def arg_setting():    
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=False)

    group.add_argument('--vae_path',
                       type=str,
                       help='path to your trained discrete VAE')

    group.add_argument('--dalle_path',
                       type=str,
                       help='path to your partially trained DALL-E')

    parser.add_argument(
        '--image_text_folder',
        type=str,
        default='/opt/ml/input/data/training',
        help='path to your folder of images and text for learning the DALL-E')


    parser.add_argument(
        '--truncate_captions',
        dest='truncate_captions',
        action='store_true',
        help=
        'Captions passed in which exceed the max token length will be truncated if this is set.'
    )

    parser.add_argument('--random_resize_crop_lower_ratio',
                        dest='resize_ratio',
                        type=float,
                        default=0.75,
                        help='Random resized crop lower ratio')

    parser.add_argument('--chinese', dest='chinese', action='store_true')

    parser.add_argument('--taming', 
                        dest='taming',
                        type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                        default=False
#                         action='store_true'
                       )

    parser.add_argument('--hug', dest='hug', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False)

    parser.add_argument('--bpe_path',
                        type=str,
                        help='path to your huggingface BPE json file')

    parser.add_argument(
        '--fp16',
        type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
        default=False,
#         action='store_true',
        help='(experimental) - Enable DeepSpeed 16 bit precision. Reduces VRAM.')

    parser.add_argument(
        '--wandb_name',
        default='dalle_train_transformer',
        help=
        'Name W&B will use when saving results.\ne.g. `--wandb_name "coco2017-full-sparse"`'
    )

    parser = distributed_utils.wrap_arg_parser(parser)

    train_group = parser.add_argument_group('Training settings')

    train_group.add_argument('--epochs',
                             default=20,
                             type=int,
                             help='Number of epochs')

    train_group.add_argument('--batch_size',
                             default=4,
                             type=int,
                             help='Batch size')

    train_group.add_argument('--learning_rate',
                             default=3e-4,
                             type=float,
                             help='Learning rate')

    train_group.add_argument('--clip_grad_norm',
                             default=0.5,
                             type=float,
                             help='Clip gradient norm')

    train_group.add_argument('--lr_decay', dest='lr_decay', action='store_true')

    model_group = parser.add_argument_group('Model settings')

    model_group.add_argument('--dim', default = 512, type = int, help = 'Model dimension')
              
    model_group.add_argument('--heads',
                             default=8,
                             type=int,
                             help='Model number of heads')

    model_group.add_argument('--dim_head',
                             default=64,
                             type=int,
                             help='Model head dimension')

    model_group.add_argument('--reversible',
                             dest='reversible',
                             type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                             default=False
#                              action='store_true'
                            )

    model_group.add_argument('--loss_img_weight',
                             default=7,
                             type=int,
                             help='Image loss weight')

    model_group.add_argument('--text_seq_len',
                             default=256,
                             type=int,
                             help='Text sequence length')

    model_group.add_argument('--depth', default=2, type=int, help='Model depth')


    model_group.add_argument(
        '--attn_types',
        default='full',
        type=str,
        help=
        'comma separated list of attention types. attention type can be: full or sparse or axial_row or axial_col or conv_like.'
    )

    
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--model_dir', type=str, default='model/dalle/')

    parser.add_argument('--num-gpus', type=int, default=8)

    parser.add_argument('--hosts', type=list, default=[])
    parser.add_argument('--current-host', type=str, default="")
    parser.add_argument(
        '--backend',
        type=str,
        default='nccl',
        help=
        'backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)'
    )

    
    
    # Setting for Model Parallel   
    parser.add_argument("--sagemakermp", type=lambda s:s.lower() in ['true','t','yes','1'], default=False)
    parser.add_argument("--num_microbatches", type=int, default=4)
    parser.add_argument("--num_partitions", type=int, default=2)
    parser.add_argument("--horovod", type=bool, default=False)
    parser.add_argument("--ddp", type=bool, default=True)
    parser.add_argument("--amp", type=int, default=0)  ## if amp is 1, true 
    parser.add_argument("--pipeline", type=str, default="interleaved")
    parser.add_argument("--optimize", type=str, default="speed")
    parser.add_argument("--placement_strategy", type=str, default="spread")
    parser.add_argument("--assert-losses", type=bool, default=False)
    
    parser.add_argument('--mp_parameters', type=str, default='')
    parser.add_argument("--partial-checkpoint",
                        type=str,
                        default="",
                        help="The checkpoint path to load")
    parser.add_argument("--full-checkpoint",
                        type=str,
                        default="",
                        help="The checkpoint path to load")
    parser.add_argument("--save-full-model",
                        action="store_true",
                        default=False,
                        help="For Saving the current Model")
    parser.add_argument(
        "--save-partial-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )

    args = parser.parse_args()
    return args


# # quit early if you used the wrong folder name
# assert Path(args.image_text_folder).exists(), f'The path {args.image_text_folder} was not found.'


def check_sagemaker(args):
    ## SageMaker

    try:
        logger.debug(
            f"os.environ.get('SM_CHANNEL_TRAINING') : {os.environ.get('SM_CHANNEL_TRAINING')}"
        )
        logger.debug(
            f"os.environ['SM_MODEL_DIR'] : {os.environ['SM_MODEL_DIR']}")
        logger.debug(
            f"os.environ['SM_NUM_GPUS'] : {os.environ['SM_NUM_GPUS']}")
        logger.debug(
            f"json.loads(os.environ['SM_HOSTS']) : {json.loads(os.environ['SM_HOSTS'])}"
        )
        logger.debug(
            f" os.environ['SM_CURRENT_HOST'] : { os.environ['SM_CURRENT_HOST']}"
        )

        if os.environ.get('SM_CHANNEL_TRAINING') is not None:
            args.image_text_folder = os.environ.get('SM_CHANNEL_TRAINING')
            model_path = '/opt/ml/code'
            #     args.vae_path = 'model/vae/vae-final.pt'
            #     args.vae_path = os.path.join(model_path, args.vae_path)
            #     args.dalle_path = os.path.join(model_path, args.dalle_path)
            args.model_dir = os.environ['SM_MODEL_DIR']
            args.num_gpus = int(os.environ['SM_NUM_GPUS'])
            args.hosts = json.loads(os.environ['SM_HOSTS'])
            args.current_host = os.environ['SM_CURRENT_HOST']
            
            args.job_name = json.loads(os.environ['SM_TRAINING_ENV'])['job_name']
            
            # uncorrectable NVLink error was detected during the execution
            os.environ["NCCL_IB_DISABLE"]="1"
            

    except:
        logger.debug("Not SageMaker")
        pass
    wandb_setting()
    return args


def wandb_setting():
    set_path = '/opt/ml/code/.netrc'
    file = Path(set_path)
    if file.exists():
        subprocess.run(['cp', '-r', set_path, '/root/.netrc'])


# helpers


def exists(val):
    return val is not None


def save_model(args, path):
    save_obj = {
        'hparams': args.dalle_params,
        'vae_params': args.vae_params,
    }
    if args.using_deepspeed:
        cp_dir = cp_path_to_dir(path, 'ds')

        args.distr_dalle.save_checkpoint(cp_dir, client_state=save_obj)

        # if not distr_backend.is_root_worker():
        if args.rank == 0:
            return

        # Save auxiliary values so we can reuse the standard routine
        # for loading.
        save_obj = {
            **save_obj,
            # Save a nonsense value that directs the user to
            # further help.
            'weights': ('To get a working standard checkpoint, '
                        'look into consolidating DeepSpeed checkpoints.'),
        }
        torch.save(save_obj, str(cp_dir / args.DEEPSPEED_CP_AUX_FILENAME))
        return
    
    elif args.sagemakermp:
        if smp.dp_rank == 0:
            if args.save_full_model:
                model_dict = args.distr_dalle.state_dict()
                opt_dict = args.distr_opt.state_dict()
                smp.save(
                    {
                        "model_state_dict": model_dict,
                        "optimizer_state_dict": opt_dict
                    },
                    path,
                    partial=False,
                )
            else:
                model_dict = args.distr_dalle.local_state_dict()
                opt_dict = args.distr_opt.local_state_dict()
                smp.save(
                    {
                        "model_state_dict": model_dict,
                        "optimizer_state_dict": opt_dict
                    },
                    path,
                    partial=True,
                )        
        smp.barrier()
    # if not distr_backend.is_root_worker():
    if args.rank == 0:
        return

    save_obj = {**save_obj, 'weights': args.dalle.state_dict()}

    torch.save(save_obj, path)


def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]


# helpers


def group_weight(model):
    group_decay, group_no_decay = [], []
    for params in model.named_parameters():
        if 'transformer' in params[0]:
            if 'bias' in params[0] or 'norm' in params[0]:
                group_no_decay.append(params[1])
                continue
        group_decay.append(params[1])

    assert len(list(
        model.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [
        dict(params=group_decay),
        dict(params=group_no_decay, weight_decay=.0)
    ]
    return groups


def cp_path_to_dir(cp_path, tag):
    """Convert a checkpoint path to a directory with `tag` inserted.
    If `cp_path` is already a directory, return it unchanged.
    """
    if not isinstance(cp_path, Path):
        cp_path = Path(cp_path)
    if cp_path.is_dir():
        return cp_path
    path_sans_extension = cp_path.parent / cp_path.stem
    cp_dir = Path(f'{path_sans_extension}-{tag}-cp')
    return cp_dir


def smp_init(args):
    args.deepspeed = False
    cfg = {
        "microbatches": args.num_microbatches,
        "placement_strategy": args.placement_strategy,
        "pipeline": args.pipeline,
        "optimize": args.optimize,
        "partitions": args.num_partitions,
        "horovod": args.horovod,
        "ddp": args.ddp,
    }

    smp.init(cfg)

    args.rank = smp.dp_rank()
    args.global_rank = smp.rank()
    args.world_size = smp.size()
    
    
    os.environ['RANK'] = str(args.rank)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['LOCAL_RANK'] = str(smp.local_rank())
    
#     ## SMP_SKIP_GRAPH_VALIDATION=1
    os.environ['SMP_SKIP_GRAPH_VALIDATION'] = "0"
    
#     args.bpe_path = "/opt/ml/code/dalle_pytorch/data/bpe_simple_vocab_16e6.txt"
    
    torch.cuda.set_device(smp.local_rank())
    args.local_rank = smp.local_rank()
    
#     if args.seed is not None:
#         random.seed(args.seed)
#         torch.manual_seed(args.seed+args.rank)
#         np.random.seed(args.seed)
#         torch.cuda.manual_seed_all(args.seed)
        
#     cudnn.deterministic = True

#     if cudnn.deterministic:
#         warnings.warn('You have chosen to seed training. '
#                       'This will turn on the CUDNN deterministic setting, '
#                       'which can slow down your training considerably! '
#                       'You may see unexpected behavior when restarting '
#                       'from checkpoints.')
    return args


def split_dataset(args):
    if (args.ddp or args.horovod) and smp.dp_size() > 1:
        partitions_dict = {f"{i}": 1 / smp.dp_size() for i in range(smp.dp_size())}
        args.ds = SplitDataset(args.ds, partitions=partitions_dict)
        args.ds.select(f"{smp.dp_rank()}")
    return args.ds


# # Rubik: Define smp.step. Return any tensors needed outside.
@smp.step()
def train_step(args, text, images, return_loss=True):
    with autocast(enabled = (args.amp > 0)):
        loss = args.distr_dalle(text, images, return_loss, args)

    scaled_loss = scaler.scale(loss) if args.amp else loss
    args.distr_dalle.backward(scaled_loss)
    return loss



def main(args):
    # constants

    VAE_PATH = args.vae_path
    DALLE_PATH = args.dalle_path
    RESUME = exists(DALLE_PATH)

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    LEARNING_RATE = args.learning_rate
    GRAD_CLIP_NORM = args.clip_grad_norm
    LR_DECAY = args.lr_decay

    MODEL_DIM = args.dim
    TEXT_SEQ_LEN = args.text_seq_len
    DEPTH = args.depth
    HEADS = args.heads
    DIM_HEAD = args.dim_head
    REVERSIBLE = args.reversible
    LOSS_IMG_WEIGHT = args.loss_img_weight

    ATTN_TYPES = tuple(args.attn_types.split(','))

    DEEPSPEED_CP_AUX_FILENAME = 'auxiliary.pt'

    # initialize distributed backend

    # initialize distributed backend
    if args.sagemakermp:
        args.deepspeed = False
        using_deepspeed = False
    else:
        args.deepspeed = True
    
    distr_backend = distributed_utils.set_backend_from_args(args)
    distr_backend.initialize(args)

    if args.sagemakermp:
        args = smp_init(args)
        distributed_utils.using_backend(distributed_utils.SageMakerMPBackend)
    else:
        using_deepspeed = \
            distributed_utils.using_backend(distributed_utils.DeepSpeedBackend) 
        args.rank = int(os.environ.get('RANK'))
        args.world_size = int(os.environ.get('WORLD_SIZE'))
        args.local_rank = int(os.environ.get('LOCAL_RANK'))
        args.global_rank = args.rank
    
    logger.debug(f"using_deepspeed : {using_deepspeed}")
    
    logger.debug(
        f"args.local_rank : {args.local_rank}, args.rank : {args.rank}")
    

    # tokenizer
    logger.debug(f"exists(args.bpe_path) : {exists(args.bpe_path)}, args.chinese : {args.chinese}")
    if exists(args.bpe_path):
        klass = HugTokenizer if args.hug else YttmTokenizer
        tokenizer = klass(args.bpe_path)
    elif args.chinese:
        tokenizer = ChineseTokenizer()
    else:
        tokenizer = SimpleTokenizer()

    # reconstitute vae

    if RESUME:
        dalle_path = Path(DALLE_PATH)
        if using_deepspeed:
            cp_dir = cp_path_to_dir(dalle_path, 'ds')
            assert cp_dir.is_dir(), \
                f'DeepSpeed checkpoint directory {cp_dir} not found'
            dalle_path = cp_dir / DEEPSPEED_CP_AUX_FILENAME
        else:
            assert dalle_path.exists(), 'DALL-E model file does not exist'
        loaded_obj = torch.load(str(dalle_path), map_location='cpu')

        dalle_params, vae_params, weights = loaded_obj['hparams'], loaded_obj[
            'vae_params'], loaded_obj['weights']

        if vae_params is not None:
            vae = DiscreteVAE(**vae_params)
        else:
            vae_klass = OpenAIDiscreteVAE if not args.taming else VQGanVAE1024
            vae = vae_klass(args)

        dalle_params = dict(**dalle_params)
        IMAGE_SIZE = vae.image_size
    else:
        if exists(VAE_PATH):
            vae_path = Path(VAE_PATH)
            assert vae_path.exists(), 'VAE model file does not exist'
            assert not vae_path.is_dir(), \
                ('Cannot load VAE model from directory; please use a '
                'standard *.pt checkpoint. '
                'Currently, merging a DeepSpeed-partitioned VAE into a DALLE '
                'model is not supported.')

            loaded_obj = torch.load(str(vae_path))

            vae_params, weights = loaded_obj['hparams'], loaded_obj['weights']

            vae = DiscreteVAE(**vae_params)
            vae.load_state_dict(weights)
        else:
            if args.rank == 0:
                # if distr_backend.is_root_worker():
                print('using pretrained VAE for encoding images to tokens')
            vae_params = None
            logger.debug(f"************* args.taming : {args.taming}")
            vae_klass = OpenAIDiscreteVAE if not args.taming else VQGanVAE1024
            vae = vae_klass(args)

        IMAGE_SIZE = vae.image_size

        dalle_params = dict(
            num_text_tokens=tokenizer.vocab_size,
            text_seq_len=TEXT_SEQ_LEN,
            dim=MODEL_DIM,
            depth=DEPTH,
            heads=HEADS,
            dim_head=DIM_HEAD,
            reversible=REVERSIBLE,
            loss_img_weight=LOSS_IMG_WEIGHT,
            attn_types=ATTN_TYPES,
        )

    # configure OpenAI VAE for float16s

    if isinstance(vae, OpenAIDiscreteVAE) and args.fp16:
        vae.enc.blocks.output.conv.use_float16 = True

    # create dataset and dataloader

    is_shuffle = not distributed_utils.using_backend(
        distributed_utils.HorovodBackend)

    ds = TextImageDataset(
        args.image_text_folder,
        text_len=TEXT_SEQ_LEN,
        image_size=IMAGE_SIZE,
        resize_ratio=args.resize_ratio,
        truncate_captions=args.truncate_captions,
        tokenizer=tokenizer,
        shuffle=is_shuffle,
    )

    assert len(ds) > 0, 'dataset is empty'
    # if distr_backend.is_root_worker():
    if args.rank == 0:
        print(f'{len(ds)} image-text pairs found for training')

    if not is_shuffle:
        data_sampler = torch.utils.data.distributed.DistributedSampler(
            ds, num_replicas=args.world_size, rank=args.rank)
    elif args.sagemakermp:
        args.ds = ds
        ds = split_dataset(args)
        data_sampler = None
    else:
        data_sampler = None
        
        
    print(f"data_sampler : {data_sampler}")

    # uncorrectable NVLink error was detected during the execution  --> remove
    kwargs = {'num_workers': args.num_worker, 'pin_memory': True}
    dl = DataLoader(ds,
                    batch_size=BATCH_SIZE,
                    shuffle=is_shuffle,
                    drop_last=True,
                    sampler=data_sampler,
                    **kwargs
                   )
    
    logger.info("Processes {}/{} ({:.0f}%) of train data".format(
    len(dl.sampler), len(dl.dataset),
    100. * len(dl.sampler) / len(dl.dataset)))

    # initialize DALL-E
    dalle = DALLE(vae=vae, **dalle_params)
    if not using_deepspeed:
        if args.fp16:
            dalle = dalle.half()
        dalle = dalle.cuda()

    if RESUME and not using_deepspeed:
        dalle.load_state_dict(weights)

    # optimizer
    opt = Adam(get_trainable_params(dalle), lr=LEARNING_RATE)
    

    if LR_DECAY:
        scheduler = ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=10,
            cooldown=10,
            min_lr=1e-6,
            verbose=True,
        )
    # if distr_backend.is_root_worker():
    if args.global_rank == 0:
        # experiment tracker

        model_config = dict(depth=DEPTH, heads=HEADS, dim_head=DIM_HEAD)
        
        
        logger.debug(f"args.wandb_name : {args.wandb_name}, RESUME : {RESUME}")
        
        run = wandb.init(
            project=args.wandb_name,  # 'dalle_train_transformer' by default
            resume=RESUME,
            config=model_config,
        )

    # distribute
    distr_backend.check_batch_size(BATCH_SIZE)
    deepspeed_config = {
        'train_batch_size': BATCH_SIZE,
        'gradient_clipping': GRAD_CLIP_NORM,
        'fp16': {
            'enabled': args.fp16,
        },
    }
    (distr_dalle, distr_opt, distr_dl,
     distr_scheduler) = distr_backend.distribute(
         args=args,
         model=dalle,
         optimizer=opt,
         model_parameters=get_trainable_params(dalle),
         training_data=ds if using_deepspeed else dl,
         lr_scheduler=scheduler if LR_DECAY else None,
         config_params=deepspeed_config,
     )
    avoid_model_calls = using_deepspeed and args.fp16

    if args.sagemakermp:
        args.distr_dalle = smp.DistributedModel(distr_dalle)
        args.scaler = smp.amp.GradScaler()
        args.distr_opt = smp.DistributedOptimizer(distr_opt)
    
    if RESUME and using_deepspeed:
        distr_dalle.load_checkpoint(str(cp_dir))
        


    # training

    for epoch in range(EPOCHS):
        logger.debug(f"********* epoch : {epoch} **********")
        if data_sampler:
            data_sampler.set_epoch(epoch)
            
        for i, (text, images) in enumerate(distr_dl):
            if args.fp16:
                images = images.half()
                
            text, images = map(lambda t: t.cuda(), (text, images))
            
            if args.sagemakermp:
                args.distr_opt.zero_grad()
                
                loss = train_step(args, text, images, return_loss=True)
                loss = loss.reduce_mean()

            else:  
                loss = distr_dalle(text, images, return_loss=True, args=args)

            if using_deepspeed:
                distr_dalle.backward(loss)
                distr_dalle.step()
                # Gradients are automatically zeroed after the step
            elif args.sagemakermp:
                if args.amp:
                    scaler.step(args.distr_opt)
                    scaler.update()
                else:
                    # some optimizers like adadelta from PT 1.8 dont like it when optimizer.step is called with no param
                    if len(list(args.distr_dalle.local_parameters())) > 0:
                        args.distr_opt.step()
            else:
                loss.backward()
                clip_grad_norm_(distr_dalle.parameters(), GRAD_CLIP_NORM)
                distr_opt.step()
                distr_opt.zero_grad()

            # Collective loss, averaged
            avg_loss = distr_backend.average_all(loss)

            log = {}

            # if i % 10 == 0 and distr_backend.is_root_worker():
            if i % 10 == 0 and args.rank == 0:
                print(epoch, i, f'loss - {avg_loss.item()}')

                log = {
                    **log, 'epoch': epoch,
                    'iter': i,
                    'loss': avg_loss.item()
                }

            if i % 100 == 0:
                # if distr_backend.is_root_worker():
                if args.rank == 0:
                    sample_text = text[:1]
                    token_list = sample_text.masked_select(
                        sample_text != 0).tolist()
                    decoded_text = tokenizer.decode(token_list)
                    
                    logger.debug(f"******* avoid_model_calls : {avoid_model_calls}")
                    if not avoid_model_calls:
                        # CUDA index errors when we don't guard this
                        image = dalle.generate_images(
                            text[:1], filter_thres=0.9)  # topk sampling at 0.9

                    wandb.save(f'./dalle.pt')

                    log = {
                        **log,
                    }
                    if not avoid_model_calls:
                        log['image'] = wandb.Image(image, caption=decoded_text)

                args.distr_dalle = distr_dalle
                args.dalle_params = dalle_params
                args.vae_params = vae_params
                args.using_deepspeed = using_deepspeed
                args.DEEPSPEED_CP_AUX_FILENAME = DEEPSPEED_CP_AUX_FILENAME

                save_model(args, f'{args.model_dir}/dalle.pt')

            # if distr_backend.is_root_worker():
            if args.rank == 0:
                wandb.log(log)
                
#             text, images = prefetcher.next()

        if LR_DECAY and not using_deepspeed:
            # Scheduler is automatically progressed after the step when
            # using DeepSpeed.
            distr_scheduler.step(loss)

        # if distr_backend.is_root_worker():
        if args.global_rank == 0:
            # save trained model to wandb as an artifact every epoch's end

            model_artifact = wandb.Artifact('trained-dalle',
                                            type='model',
                                            metadata=dict(model_config))
            # model_artifact.add_file('dalle.pt')
            run.log_artifact(model_artifact)

    args.distr_dalle = distr_dalle
    args.dalle_params = dalle_params
    args.vae_params = vae_params
    args.using_deepspeed = using_deepspeed
    args.DEEPSPEED_CP_AUX_FILENAME = DEEPSPEED_CP_AUX_FILENAME

    save_model(args, f'{args.model_dir}/dalle-final.pt')

    # if distr_backend.is_root_worker():
    if args.global_rank == 0:
        wandb.save('./dalle-final.pt')
        model_artifact = wandb.Artifact('trained-dalle',
                                        type='model',
                                        metadata=dict(model_config))
        # model_artifact.add_file('dalle-final.pt')
        run.log_artifact(model_artifact)

        wandb.finish()


if __name__ == '__main__':
    args = arg_setting()
    args = check_sagemaker(args)
    main(args)
