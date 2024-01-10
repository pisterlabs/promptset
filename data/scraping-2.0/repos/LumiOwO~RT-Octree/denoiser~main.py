import torch

import os

from denoiser.utils import seed_everything
from denoiser.runner import Runner
from denoiser.dataset import BlenderDataset, TanksAndTemplesDataset, LLFFDataset

from denoiser.logger.base_logger import BaseLogger
from denoiser.logger.wandb_logger import WandbLogger

from denoiser.network import GuidanceNet

import configargparse

def main(args):
    # Init
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CPU-only not supported."

    # Logger
    if args.use_wandb:
        logger = WandbLogger(args)
    else:
        logger = BaseLogger(args)

    # Create model
    model = GuidanceNet(
        args.in_channels, 
        args.mid_channels, 
        args.num_branches, 
        args.num_layers, 
        args.kernel_levels
        )
    if args.task == "compact":
        runner = Runner(args, logger=logger, device=device)
        runner.compact(model)
        return
    model = model.to(device)

    # Load data
    if args.dataset_type == "blender":
        dataset = BlenderDataset(args, device=device)
    elif args.dataset_type == "tt":
        dataset = TanksAndTemplesDataset(args, device=device)
    elif args.dataset_type == "llff":
        dataset = LLFFDataset(args, device=device)
    else:
        raise NotImplementedError(f"Invalid dataset type: {args.dataset_type}.")
    logger.print("Dataset loaded.")

    # Create runner
    runner = Runner(args, dataset=dataset, logger=logger, device=device)
    if args.task == "train":
        runner.train(model)
    elif args.task == "test":
        runner.test(model)
    else:
        raise NotImplementedError(f"Invalid task type: {args.task}.")


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, 
                        help="config file path")
    parser.add_argument("--task", type=str, choices=["train", "test", "compact"],
                        help="task type")
    parser.add_argument("--logs_root", type=str, default="../logs/", 
                        help="root dir of all experiment logs")
    parser.add_argument("--exp_name", type=str, 
                        help="experiment name")
    parser.add_argument("--data_dir", type=str, default="../data/nerf_synthetic/lego", 
                        help="input data directory")

    # dataset options
    parser.add_argument("--dataset_type", type=str, default="blender", 
                        help="options: llff / blender / tt")
    parser.add_argument("--spp", type=int, default=1, 
                        help="use which spp noisy input")
    parser.add_argument("--preload", action="store_true", 
                        help="preload dataset to cuda")
    parser.add_argument("--nx", type=int, default=1, 
                        help="number of slices in x axis")
    parser.add_argument("--ny", type=int, default=1, 
                        help="number of slices in y axis")

    # logging options
    parser.add_argument("--use_wandb", action="store_true",  
                        help="use wandb dashboard")
    parser.add_argument("--i_print",   type=int, default=1, 
                        help="frequency of console printout and metric loggin")
    parser.add_argument("--i_save", type=int, default=100, 
                        help="frequency of weight ckpt saving")
    parser.add_argument("--i_test", type=int, default=100, 
                        help="frequency of testset saving")
    parser.add_argument("--save_image", action="store_true",
                        help="save test images")

    # training options
    parser.add_argument("--in_channels", type=int, default=8, 
                        help="input channels of GuidanceNet")
    parser.add_argument("--mid_channels", type=int, default=8, 
                        help="middle channels of GuidanceNet")
    parser.add_argument("--num_layers", type=int, default=8, 
                        help="number of RepVGG blocks in GuidanceNet")
    parser.add_argument("--num_branches", type=int, default=3, 
                        help="number of branches in RepVGG blocks")
    parser.add_argument("--kernel_levels", type=int, default=8, 
                        help="number of filtering levels for denoising")
    parser.add_argument("--loss_fn", type=str, default="smape", 
                        help="loss function")
    parser.add_argument("--lr", type=float, default=5e-4, 
                        help="learning rate")
    parser.add_argument('--epochs', type=int, default=30000, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="batch_size for dataloader, only valid in training")


    # args preprocess
    args = parser.parse_args()
    if args.task != "train":
        args.use_wandb = False
    args.work_dir = os.path.join(args.logs_root, args.exp_name)

    main(args)