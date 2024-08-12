import os
import wandb
import argparse
from tqdm import tqdm

from deepspeed.profiling.flops_profiler import get_model_profile

import warnings

warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.utils import loss
from src.utils.constants import NUM_CLASS
from src.utils.lr_scheduler import LinearWarmupCosineAnnealingLR

from src.dataset.dataloader import get_loader

from src.model.Dynamic_LGTransformer import Dynamic_LocalGlobal


torch.multiprocessing.set_sharing_strategy("file_system")


def train(args, train_loader, model, optimizer, loss_seg):
    model.train()

    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    ave_losses = [0] * len(loss_seg.names_losses)
    iterator = ""
    for loss in loss_seg.names_losses:
        iterator += f"{loss}=%2.5f,"

    for step, batch in enumerate(epoch_iterator):
        x, y, task = (
            batch["image"].to(args.device),
            batch["post_label"].float().to(args.device)[:, 1:],
            batch["task"],
        )

        # Calculate the upper corner coordinates of the patch
        corner = batch["crop_center"] - (torch.tensor(x.shape[2:]) / 2)
        corner = corner / batch["image_meta_dict"]["spatial_shape"]
        corner = corner.to(args.device) * 100
        spacing = torch.diagonal(batch["image_meta_dict"]["affine"], dim1=1, dim2=2)
        spacing = torch.abs(spacing)[:, :3].to(args.device)

        # Forward pass
        metadata = {"task": task, "corner": corner, "spacing": spacing}
        logit_map, features = model([x, metadata])

        # Loss
        losses = loss_seg(logit_map, y, features, task)
        total_loss = 0
        ind_losses = []
        for idx, loss in enumerate(losses):
            total_loss += losses[loss] * loss_seg.names_losses[loss]  # loss * weight
            ind_losses.append(losses[loss].item())
            ave_losses[idx] += losses[loss].item()

        # Backward pass
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log
        description = f"Epoch = %d: Training (%d / %d Steps) ({iterator})"
        epoch_iterator.set_description(
            description
            % (
                args.epoch,
                step,
                len(train_loader),
                *ind_losses,
            )
        )
        torch.cuda.empty_cache()

    header = f"Epoch = %d: Average losses - {iterator}"
    ave_losses = [ave / len(epoch_iterator) for ave in ave_losses]
    print(
        header
        % (
            args.epoch,
            *ave_losses,
        )
    )

    return ave_losses


def process(args):
    rank = 0

    if args.dist:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = args.local_rank
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)

    if rank == 0:
        config = {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
        }

        os.makedirs(os.path.join("./out", args.log_name, "wandb"), exist_ok=True)
        wd_logger = wandb.init(
            # project="DynLocGlob",
            project="CLIP-stuff",
            name=args.log_name,
            dir=os.path.join("./out", args.log_name),
            config=config,
            resume=True,
        )
        args.wandb_id = wd_logger.id  # TODO
    else:
        wd_logger = None

    # prepare the 3D model
    model = Dynamic_LocalGlobal(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=1,
        out_channels=NUM_CLASS if args.num_class == 0 else args.num_class,
        entities=args.entities,
    )

    # Load pre-trained weights
    if args.pretrain is not None:
        try:
            model.load_params(
                torch.load(args.pretrain, map_location="cpu")["state_dict"]
            )
            print("Pretrained model loaded - strict")
        except:
            model_dict = torch.load(args.pretrain, map_location="cpu")["net"]
            store_dict = model.state_dict()
            for key in model_dict.keys():
                n_key = key
                if key.startswith("module."):
                    n_key = key[7:]

                if ("backbone." + n_key) in store_dict.keys():
                    n_key = "backbone." + n_key

                store_dict[n_key] = model_dict[key]

                if (
                    "organ_embedding" in key
                    or "output" in key
                    or "controller" in key
                    or "precls_conv" in key
                ):
                    if n_key not in model.state_dict().keys():
                        continue
                    if model.state_dict()[n_key].shape != model_dict[key].shape:
                        del store_dict[n_key]
                        print(n_key, "-> deleted")

            load_resume = model.load_state_dict(store_dict, strict=False)
            print("Pretrained model loaded - no strict: ", load_resume)

    # Load the word embedding
    word_embedding = torch.load(args.word_embedding, map_location="cpu")
    model.organ_embedding = word_embedding["text_features"].float()
    model.organ_embed_idxs = word_embedding["task_indexes"]
    print("load word embedding:", args.word_embedding)

    model.to(args.device)

    # get dataloaders
    train_loader, train_sampler = get_loader(args)

    # calculate the model's flops and size
    if args.flops:
        flops, macs, params = get_model_profile(
            model=model,
            input_shape=(
                2,
                1,
                96,
                96,
                96,
            ),  # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
            args=None,  # list of positional arguments to the model.
            kwargs=None,  # dictionary of keyword arguments to the model.
            print_profile=True,  # prints the model graph with the measured profile attached to each module
            detailed=True,  # print the detailed profile
            module_depth=-1,  # depth into the nested modules, with -1 being the inner most modules
            top_modules=1,  # the number of top modules to print aggregated profile
            warm_up=10,  # the number of warm-ups before measuring the time of each module
            as_string=True,  # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
            output_file=os.path.join(
                "./out", args.log_name, "profiler.txt"
            ),  # path to the output file. If None, the profiler prints to stdout.
            ignore_modules=None,
        )  # the list of modules to ignore in the profiling
        print("FLOPs:", flops, "MACs:", macs, "Params", params)

    model.train()
    if args.dist:
        model = DistributedDataParallel(
            model, device_ids=[args.device], find_unused_parameters=True
        )

    # criterion and optimizer
    loss_seg = loss.SegmentationLoss_batch(
        w_bce=args.w_bce,
        w_dice=args.w_dice,
        w_xent=args.w_xent,
        prototypes=args.prototypes,
        multiple_pos=args.multiple_positives,
        entities=args.entities,
    ).to(args.device)

    backbone = []
    decoder = []
    others = []
    for name, param in model.named_parameters():
        if "swinViT" in name or "encoder" in name:  # backbone and skip connections
            backbone.append(param)
            if args.freeze_backbone:
                param.requires_grad = False
        elif "decoder" in name:  # decoder convs
            decoder.append(param)
            if args.freeze_decoder:
                param.requires_grad = False
        else:  # output heads, controller, text_to_vision
            others.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": others},
            {"params": decoder, "lr": args.lr * args.lr_dec_mult},  # 0.1
            {"params": backbone, "lr": args.lr * args.lr_enc_mult},  # 0.01
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=args.warmup_epoch,
        max_epochs=args.max_epoch,
        eta_min=args.min_lr,
        warmup_start_lr=args.min_lr,
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if args.dist:
            model.load_state_dict(checkpoint["net"])
        else:
            store_dict = model.state_dict()
            model_dict = checkpoint["net"]
            for key in model_dict.keys():
                if key.startswith("module."):
                    store_dict[".".join(key.split(".")[1:])] = model_dict[key]
                else:
                    store_dict[key] = model_dict[key]
            model.load_state_dict(store_dict)
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except:
            print("optimizer not loaded")
        args.epoch = checkpoint["epoch"] + 1
        scheduler.load_state_dict(checkpoint["scheduler"])

        print("successful resume from ", args.resume)

    torch.backends.cudnn.benchmark = True

    if rank == 0:
        wd_logger.watch(model, loss_seg, log="all")

    # start training
    while args.epoch < args.max_epoch:
        if args.dist:
            dist.barrier()
            train_sampler.set_epoch(args.epoch)
        scheduler.step()

        losses = train(args, train_loader, model, optimizer, loss_seg)
        wd_losses = {}
        for name, value in zip(loss_seg.names_losses, losses):
            wd_losses["train_loss_" + name] = value
        if rank == 0:
            wd_logger.log(
                {
                    "epoch": args.epoch,
                    **wd_losses,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        if args.dist:
            dist.barrier()
        if rank == 0:
            checkpoint = {
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": args.epoch,
                "wandb_id": args.wandb_id,
            }

            if not os.path.isdir("out/" + args.log_name):
                os.mkdir("out/" + args.log_name)
            torch.save(
                checkpoint,
                "out/" + args.log_name + "/epoch_last.pth",
            )
            if args.epoch % args.store_num == 0 and args.epoch != 0:
                torch.save(
                    checkpoint,
                    "out/" + args.log_name + "/epoch_" + str(args.epoch) + ".pth",
                )
            print("save model successful")

        args.epoch += 1

    if args.dist:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument(
        "--dist",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="distributed training or not",
    )
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=1)
    ## logging
    parser.add_argument(
        "--log_name", default="swinunetr", help="The path resume from checkpoint"
    )
    parser.add_argument(
        "--flops",
        action="store_true",
        default=False,
        help="only measure the model FLOPs",
    )
    ## model load
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=False,
        help="freeze the backbone",
    )
    parser.add_argument(
        "--freeze_decoder",
        action="store_true",
        default=False,
        help="freeze the decoder",
    )
    parser.add_argument(
        "--num_class", default=0, type=int, help="Number of training epoches"
    )
    parser.add_argument(
        "--entities",
        action="store_true",
        default=False,
        help="Binary segmentation of entities",
    )
    parser.add_argument(
        "--resume", default=None, help="The path resume from checkpoint"
    )
    parser.add_argument(
        "--pretrain",
        default=None,
        help="The path of pretrain model. Eg, ./pretrained_weights/swinunetr.pth",
    )
    parser.add_argument(
        "--word_embedding",
        default="./pretrained_weights/txt_encoding.pth",
        help="The path of word embedding",
    )
    ## hyperparameter
    parser.add_argument(
        "--max_epoch", default=2000, type=int, help="Number of training epoches"
    )
    parser.add_argument(
        "--store_num", default=20, type=int, help="How often to store the model"
    )
    parser.add_argument(
        "--warmup_epoch", default=100, type=int, help="number of warmup epochs"
    )
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--min_lr", default=5e-5, type=float, help="Learning rate")
    parser.add_argument(
        "--lr_enc_mult",
        default=1.0,
        type=float,
        help="Learning rate encoder multiplier",
    )
    parser.add_argument(
        "--lr_dec_mult",
        default=1.0,
        type=float,
        help="Learning rate encoder multiplier",
    )
    parser.add_argument("--weight_decay", default=1e-5, help="Weight Decay")
    parser.add_argument("--w_bce", default=1, type=float, help="BCE loss weight")
    parser.add_argument("--w_dice", default=1, type=float, help="Dice loss weight")
    parser.add_argument("--w_xent", default=0.0, type=float, help="Xent loss weight")
    parser.add_argument(
        "--prototypes",
        action="store_true",
        default=False,
        help="whether use prototypes in xent loss",
    )
    parser.add_argument(
        "--multiple_positives",
        action="store_true",
        default=False,
        help="whether use multiple positive samples in xent loss",
    )
    ## dataset
    parser.add_argument(
        "--datasetkey",
        nargs="+",
        default=["1"],
        help="datasets to use in this run",
    )
    parser.add_argument(
        "--data_root_path",
        default="/Path/to/dataset(s)/",
        help="data root path",
    )
    parser.add_argument(
        "--data_txt_path", default="./dataset/dataset_list/", help="data txt path"
    )
    parser.add_argument("--batch_size", default=2, type=int, help="batch size")
    parser.add_argument("--num_samples", default=2, type=int, help="samples per image")
    parser.add_argument(
        "--num_workers", default=8, type=int, help="workers for DataLoader"
    )
    parser.add_argument("--zoom", default=1.0, type=float, help="zoom in Zoomd")
    parser.add_argument(
        "--a_min", default=-1000, type=float, help="a_min in ScaleIntensityRanged"
    )
    parser.add_argument(
        "--a_max", default=4500, type=float, help="a_max in ScaleIntensityRanged"
    )
    parser.add_argument(
        "--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged"
    )
    parser.add_argument(
        "--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged"
    )
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")

    parser.add_argument("--phase", default="train", help="train or validation or test")
    parser.add_argument(
        "--uniform_sample",
        action="store_true",
        default=False,
        help="whether utilize uniform sample strategy",
    )
    parser.add_argument(
        "--cache_dataset",
        action="store_true",
        default=False,
        help="whether use cache dataset",
    )
    parser.add_argument(
        "--cache_rate",
        default=0.005,
        type=float,
        help="The percentage of cached data in total",
    )

    args = parser.parse_args()

    process(args=args)


if __name__ == "__main__":
    main()
