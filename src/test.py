import os
import time
import numpy as np
from tqdm import tqdm
import nibabel as nib

import torch
import torch.nn.functional as F

from monai.transforms import Invertd

from src.dataset.dataloader import get_loader
from src.model.Dynamic_LGTransformer import Dynamic_LocalGlobal

from src.utils.utils import get_key, resample_3d, dice_score
from src.utils.sliding_window_inference import sliding_window_inference
from src.utils.constants import TEMPLATE, ORGAN_NAMES, NUM_CLASS
from src.utils.config import args


def process_one_image(model, sample, args):
    image, task_id = sample["image"], sample["task"]

    shape = sample["image_meta_dict"]["spatial_shape"].to(image.device)
    spacing = torch.diagonal(sample["image_meta_dict"]["affine"], dim1=1, dim2=2)
    spacing = torch.abs(spacing)[:, :3].to(image.device)

    metadata = {
        "task": task_id * args.batch_size,
        "shape": shape,
        "spacing": spacing,
    }

    img_shape = image.shape[2:]
    roi_size = (args.roi_x, args.roi_y, args.roi_z)
    print("Image shape:", img_shape)
    print("Batch size (image patches):", args.batch_size)

    pred, _ = sliding_window_inference(
        image,
        roi_size,
        args.batch_size,
        model,
        overlap=0.25,
        padding_mode="reflect",
        device=torch.device("cpu"),  # output device
        with_coord=True,
        metadata=metadata,  # kwargs
        amp_device=args.device,
    )

    pred = F.sigmoid(pred.cpu())
    pred_hard = pred > args.threshold
    return pred[0], pred_hard[0]


def print_metrics(pred_hard, label, organ_list):
    for o_id, organ in enumerate(organ_list):
        if torch.sum(label[organ - 1, :, :, :].cuda()) != 0:
            dice_organ, recall, precision = dice_score(
                pred_hard[o_id, :, :, :].cuda(),
                label[organ - 1, :, :, :].cuda(),
            )
            print(
                "%s:\t dice %.4f,\t recall %.4f,\t precision %.4f."
                % (
                    ORGAN_NAMES[organ],
                    dice_organ.item(),
                    recall.item(),
                    precision.item(),
                )
            )
        else:
            if torch.sum(pred_hard[o_id, :, :, :]) == 0:
                print("%s: True negative" % ORGAN_NAMES[organ])
            else:
                print("%s: False positive" % ORGAN_NAMES[organ])


def prepare_inference(device):
    args.entities = True

    torch.backends.cudnn.benchmark = True

    # prepare the model
    model = Dynamic_LocalGlobal(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=1,
        out_channels=NUM_CLASS if args.num_class == 0 else args.num_class,
        entities=args.entities,
        training=False,
    )

    # Load pre-trained weights
    store_dict = model.state_dict()
    checkpoint = torch.load(args.model_path, map_location="cpu")
    load_dict = checkpoint["net"]
    args.epoch = checkpoint["epoch"]
    args.num_samples = 1

    for key, value in load_dict.items():
        if key.startswith("module."):
            key = key[7:]
        if "organ_embedding" in key:
            # because of the NA categories, the CLIP embedding has dim 42 but
            # the model returns 48 categories
            word_embedding = torch.load(args.word_embedding, map_location="cpu")
            if key in store_dict:
                del store_dict[key]
            model.organ_embedding = word_embedding["text_features"].float()
            model.organ_embed_idxs = word_embedding["task_indexes"]
            continue

        store_dict[key] = value

    model.load_state_dict(store_dict, strict=False)

    model.to(device)
    model.eval()

    with torch.no_grad():
        model.get_test_kernels(["ToothFairy2"])
    return model


def ToothFairy_inference(image, device):
    print("===== Starting image inference =====")

    start_time = time.time()
    model = prepare_inference(device)

    _, _, H, W, D = image.shape
    shape = torch.tensor([H, W, D]).to(device)
    affine = torch.eye(4)
    affine[:2] *= -1

    sample = {
        "image": image.to(device),
        "image_meta_dict": {"spatial_shape": shape[None], "affine": affine[None]},
        "task": ["ToothFairy2"],
    }

    args.device = str(device)  # for the Autocast part
    pred, pred_hard = process_one_image(model, sample, args)

    print(" => Prediction time:", time.time() - start_time)
    mid_time = time.time()
    torch.cuda.empty_cache()

    merged_pred, _ = resample_3d(
        pred_hard,
        pred,
        sample,
        not args.simple_merge,
        skip_classes=True,
    )
    print(" => Output size:", merged_pred.shape)
    print(" => Postprocessing/merging time:", time.time() - mid_time)
    print(" ==> Total processing time:", time.time() - start_time)
    return merged_pred


def main():
    model = prepare_inference(torch.device("cuda"))

    ValLoader, val_transforms = get_loader(args)
    save_dir = os.path.join("out", args.log_name, f"test_epoch{args.epoch}")
    os.makedirs(os.path.join(save_dir), exist_ok=True)
    model.eval()

    dice_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS))  # 1st row for dice, 2nd row for count

    for batch in tqdm(ValLoader):
        start_time = time.time()
        name, task_id = batch["name"], batch["task"]

        print("===== Processing case", name[0], "=====")

        task_key = get_key(task_id[0])
        organ_list = TEMPLATE[task_key]

        output_name = os.path.join(save_dir, os.path.basename(name[0]))

        if args.store_result and os.path.exists(output_name):
            print(f"Image {name[0]} already processed.")
            continue

        batch["image"] = batch["image"].cuda()
        pred, pred_hard = process_one_image(model, batch, args)
        print(" => Inference time:", time.time() - start_time)
        mid_time = time.time()
        torch.cuda.empty_cache()

        print_metrics(pred_hard, batch["post_label"][0, 1:], organ_list)

        map_empty_classes = task_id[0] == "ToothFairy2"
        merged_pred, _ = resample_3d(
            pred_hard,
            pred,
            batch,
            not args.simple_merge,
            skip_classes=map_empty_classes,
        )
        print(" => Postprocessing/merging time:", time.time() - mid_time)

        if args.store_result:
            nib.save(
                nib.Nifti1Image(
                    merged_pred,
                    batch["image_meta_dict"]["affine"].squeeze(),
                ),
                output_name,
            )
        print(" ==> Total processing time:", time.time() - start_time)

    torch.cuda.empty_cache()

    ave_organ_dice = np.zeros((2, NUM_CLASS))

    with open("out/" + args.log_name + f"/test_{args.epoch}.txt", "w") as f:
        for key in TEMPLATE.keys():
            organ_list = TEMPLATE[key]
            content = "Task%s| " % (key)
            for organ in organ_list:
                dice = dice_list[key][0][organ - 1] / dice_list[key][1][organ - 1]
                content += "%s: %.4f, " % (ORGAN_NAMES[organ], dice)
                ave_organ_dice[0][organ - 1] += dice_list[key][0][organ - 1]
                ave_organ_dice[1][organ - 1] += dice_list[key][1][organ - 1]
            print(content)
            f.write(content)
            f.write("\n")
        content = "Average | "
        for i in range(NUM_CLASS):
            content += "%s: %.4f, " % (
                ORGAN_NAMES[i + 1],
                ave_organ_dice[0][i] / ave_organ_dice[1][i],
            )
        print(content)
        f.write(content)
        f.write("\n")
        print(np.nanmean(ave_organ_dice[0] / ave_organ_dice[1]))
        f.write(
            "%s: %.4f, "
            % ("average", np.nanmean(ave_organ_dice[0] / ave_organ_dice[1]))
        )
        f.write("\n")


if __name__ == "__main__":
    main()
