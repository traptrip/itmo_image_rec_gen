import numpy as np


def get_metrics(pred, gt):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))  
    l2 = np.sum((pred - gt)**2)
    gt_sum = np.sum(gt == 1)
    pred_sum = np.sum(pred == 1)

    precision = 1. if tp == 0 and fp == 0 else tp / (tp + fp)
    recall = tp / (tp + fn)
    iou = tp / (gt_sum + pred_sum - tp)

    return tp, fp, fn, l2, precision, recall, iou, gt_sum, pred_sum


def get_metrics_batch(pred_masks, gt_masks):
    n_samples = len(pred_masks)
    tp, fp, fn, l2, gt_sum, pred_sum = 0, 0, 0, 0, 0, 0
    statistics = []
    for pred, gt in zip(pred_masks, gt_masks):
        s = get_metrics(pred, gt)
        statistics.append(s)
        tp += s[0]
        fp += s[1]
        fn += s[2]
        l2 += s[3]
        gt_sum += s[7]
        pred_sum += s[8]
    statistics = np.array(statistics)
    l2 /= n_samples

    # Micro statistics
    precision_micro = 1. if tp == 0 and fp == 0 else tp / (tp + fp)
    recall_micro = tp / (tp + fn)
    iou_micro = tp / (gt_sum + pred_sum - tp)
    micro_stat = [precision_micro, recall_micro, l2, iou_micro]

    # Macro statistics
    precisions = statistics[:, 4]
    recalls = statistics[:, 5]
    ious = statistics[:, 6]
    precision_macro = precisions.mean()
    recall_macro = recalls.mean()
    iou_macro = ious.mean()
    macro_stat = [precision_macro, recall_macro, l2, iou_macro]

    # IoU >= 0.5, IoU >= 0.75, IoU >= 0.9
    iou_05 = np.sum(ious >= 0.5) / n_samples
    iou_075 = np.sum(ious >= 0.75) / n_samples
    iou_09 = np.sum(ious >= 0.9) / n_samples

    return micro_stat, macro_stat, iou_05, iou_075, iou_09
