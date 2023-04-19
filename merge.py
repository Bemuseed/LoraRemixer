from tqdm import tqdm

def merge(model_a, model_b, names=""):
    model_c = dict()
    for k in tqdm(model_a.keys(), "Merging models "+names+"..."):
        model_c[k] = (0.5 * model_a[k]) + (0.5 * model_b[k])
    return model_c
