import string
import re
import collections

def normalize_answer(s):
    """Lower text and remove punctuation and extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    def normalize(text):
        return " ".join(word for word in text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def add_spaces_after_punct(text):
        return re.sub(r'([.,!?;])', r'\1 ', text)

    return white_space_fix(normalize(remove_punc(add_spaces_after_punct(lower(s)))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(a_gold_list, a_pred_list):
    f1_list = []
    for a_gold, a_pred in zip(a_gold_list, a_pred_list):
        gold_toks = get_tokens(a_gold)
        pred_toks = get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            f1 = int(gold_toks == pred_toks)
            f1_list.append(f1)
            continue
        if num_same == 0:
            f1 = 0
            f1_list.append(f1)
            continue
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_list.append(f1)
    return sum(f1_list)/max(len(f1_list),1)