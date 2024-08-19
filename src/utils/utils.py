import cc3d
import fastremap
import numpy as np
import scipy.ndimage as ndimage

import torch

from src.utils.constants import MERGE_MAPPING, REVERSE_TASK_NAMES, ORGAN_NAMES


def keep_topk_largest_connected_object(npy_mask, k, area_least, out_mask, factor=0):
    labels_out = cc3d.connected_components(npy_mask, connectivity=26)
    areas = {}
    for label, extracted in cc3d.each(labels_out, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)

    if len(candidates) > 0 and candidates[0][1] < area_least:
        area_least *= 0.5

    if len(candidates) > 0 and factor > 0:
        ths = max(candidates[0][1] * factor, area_least)
    else:
        ths = area_least

    count = 1
    for i in range(min(k, len(candidates))):
        if candidates[i][1] > ths:
            out_mask[labels_out == int(candidates[i][0])] = count
            count += 1


def extract_topk_largest_candidates(npy_mask, organ_num, area_least=0, I=0, factor=0):
    ## npy_mask: w, h, d
    ## organ_num: the maximum number of connected component
    out_mask = np.zeros(npy_mask.shape, np.uint8)
    t_mask = npy_mask.copy()

    keep_topk_largest_connected_object(t_mask, organ_num, area_least, out_mask, factor)

    if I > 0:
        out_mask = ndimage.binary_closing(
            np.pad(out_mask, I) > 0,
            iterations=I,
            structure=np.ones((3, 3, 3)),
        )[I:-I, I:-I, I:-I]

    return out_mask


def add_category(final_mask, organ_label, intersection, tgt):
    # most likely is just the contours
    if intersection.sum() < 100:
        final_mask[organ_label > 0] = tgt
        return final_mask

    conflict_inst = organ_label * intersection
    ids_check = fastremap.unique(conflict_inst)[1:]
    tgt_i, tgt_size = fastremap.unique(organ_label, return_counts=True)
    tgt_i, tgt_size = tgt_i[1:], tgt_size[1:]

    # Iterate through the organ instances that have overlaping
    for iid, isize in zip(tgt_i, tgt_size):
        if iid not in ids_check:
            continue

        target_instance = organ_label == iid
        if isize < 100:
            organ_label[target_instance] = 0
            continue

        # other organs that also predict in the same voxels
        conflict_labels = fastremap.unique(final_mask[target_instance])
        for conflict in conflict_labels:
            if conflict == 0:
                continue

            # check individual instances of the conflicting label
            cft_binary = final_mask == conflict
            tgt_inst = cc3d.connected_components(cft_binary, connectivity=26)
            c_id, c_size = fastremap.unique(
                tgt_inst * target_instance, return_counts=True
            )  # overlaps with target_instance
            c_id, c_size = c_id[1:], c_size[1:]  # intersection id and size

            for i, j in zip(c_id, c_size):
                _inst = tgt_inst == i  # intersecting instance
                _size = _inst.sum()
                if j < 100:
                    # organ_label[(_inst * intersection).astype(bool)] = 0
                    continue

                # if conflicting inst is smaller than existing pred, delete it
                if _size > isize:
                    organ_label[_inst] = 0
                else:  # else, change all existing pred instance to new tgt
                    organ_label[_inst] = tgt_i[-1] + 1

        target_instance = organ_label == iid
        if target_instance.sum() < 100:
            organ_label[target_instance] = 0

    final_mask[organ_label > 0] = tgt
    return final_mask


def postprocess_results(pred_masks, mapping):
    C, W, H, D = pred_masks.shape

    minimum_side = 17  # ~5k -> smallest on average is ~8k
    fct = 1
    resize = (H * W * D) > (410 * 410 * 300)
    if resize:
        print("Downsampling image for postprocessing.")
        fct = 0.8
        pred_masks = (
            ndimage.zoom(pred_masks, (1, fct, fct, fct), order=0, prefilter=False) > 0
        ).astype(np.uint8)

    minimum_side *= fct  # too small?
    C, nW, nH, nD = pred_masks.shape
    final_mask = np.zeros((nW, nH, nD), dtype=np.uint8)

    left_mask = np.zeros((nW, nH, nD), dtype=np.uint8)
    right_mask = np.zeros((nW, nH, nD), dtype=np.uint8)
    left_mask[nW - int(nW * 0.6) :] = 1
    right_mask[: int(nW * 0.6)] = 1

    for item in mapping:
        src, tgt = item
        min_side = minimum_side
        if tgt in [8, 9, 10]:  # bridge, crown, implant
            num = 10
            min_side = minimum_side * 0.6
        elif tgt in [1, 2, 7]:  # jawbones, pharinx
            num = 1
        else:
            num = 2

        if "Left" in ORGAN_NAMES[tgt]:
            organ_label = pred_masks[src - 1] * left_mask
        elif "Right" in ORGAN_NAMES[tgt]:
            organ_label = pred_masks[src - 1] * right_mask
        else:
            organ_label = pred_masks[src - 1]

        organ_label = extract_topk_largest_candidates(
            organ_label,
            num,
            area_least=min_side**3,
            I=0,
            factor=0.2,
        )

        # Add the classes that do not intersect with anything
        intersection = (final_mask * organ_label) > 0
        if intersection.sum() == 0:
            final_mask[organ_label.astype(bool)] = tgt
            continue

        final_mask = add_category(final_mask, organ_label, intersection, tgt)

    size_filter = 25 * fct
    last_filter = extract_topk_largest_candidates(
        final_mask > 0, 5, area_least=size_filter**3, I=0
    )
    final_mask = final_mask * (last_filter > 0)
    if resize:
        factor = np.array([W, H, D]) / np.array(final_mask.shape)
        final_mask = ndimage.zoom(final_mask, factor, order=0, prefilter=False)
        final_mask = np.round(final_mask)
    return final_mask


def simple_merge(pred_masks, logits, mapping):
    _, W, H, D = pred_masks.shape

    task_logits = torch.zeros(mapping[-1][1], W, H, D)
    mask = torch.zeros(W, H, D)

    for item in mapping:
        src, tgt = item
        mask += pred_masks[src - 1]
        task_logits[tgt - 1] = logits[src - 1]

    task_merged = torch.argmax(task_logits, 0) + 1  # why so slow?
    merged_label = task_merged * (mask > 0)
    return merged_label


def resample_3d(pred_hard, logits, batch, do_postprocessing, skip_classes=True):
    template_key = get_key(batch["task"][0])
    transfer_mapping = MERGE_MAPPING[template_key]
    if skip_classes:
        transfer_mapping = [(idx + 1, i) for idx, (_, i) in enumerate(transfer_mapping)]

    # merge the labels
    if do_postprocessing:
        print(" => Postprocessing...")
        merged_pred = postprocess_results(
            pred_hard,
            transfer_mapping,
        )
        merged_pred = merged_pred.astype(np.uint8)
    else:
        print(" => Merging binary masks...")
        merged_pred = simple_merge(
            pred_hard,
            logits,
            transfer_mapping,
        )
        merged_pred = merged_pred.detach().numpy().astype(np.uint8)

    # re-orient the output to match the original input
    affine = batch["image_meta_dict"]["affine"][0]
    orientation = affine.diag().sign()[:-1]
    flip = np.where(orientation == -1)[0]

    if len(flip) >= 1:
        merged_pred = np.flip(merged_pred, flip)
    return merged_pred


def get_key(name):
    task = REVERSE_TASK_NAMES[name]
    return task


def dice_score(preds, labels, spe_sen=False):  # on GPU
    ### preds: w,h,d; label: w,h,d
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    # preds = torch.where(preds > 0.5, 1.0, 0.0)
    predict = preds.contiguous().view(1, -1)
    target = labels.contiguous().view(1, -1)

    tp = torch.sum(torch.mul(predict, target))
    fn = torch.sum(torch.mul(predict != 1, target))
    fp = torch.sum(torch.mul(predict, target != 1))
    tn = torch.sum(torch.mul(predict != 1, target != 1))

    den = torch.sum(predict) + torch.sum(target) + 1

    dice = 2 * tp / den
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (fp + tn)

    # print(dice, recall, precision)
    if spe_sen:
        return dice, recall, precision, specificity
    else:
        return dice, recall, precision
