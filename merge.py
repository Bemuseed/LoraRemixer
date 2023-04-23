from tqdm import tqdm

def merge(model_a, model_b, ratio=0.5):
    model_c = dict()
    for k in model_a.keys():
        model_c[k] = (ratio * model_a[k]) + ((1 - ratio) * model_b[k])
    return model_c
