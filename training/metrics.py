import re
from collections import Counter

from data.preprocessing import normalize_text, preprocess_sentence

def exact_match(pred, gt):
    return int(preprocess_sentence(normalize_text(pred)) == preprocess_sentence(normalize_text(gt)))

def f1_score(pred, gt, is_return_precision_recall=False):
    pred_tokens = preprocess_sentence(normalize_text(pred)).split()
    gt_tokens = preprocess_sentence(normalize_text(gt)).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    if is_return_precision_recall:
        return f1, precision, recall
    return f1
