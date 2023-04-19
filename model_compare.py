from itertools import combinations
import numpy
from tqdm import tqdm
import flat
import tensor_manager
import merge
import torch

PATHS=["vincent_di_fate-000017.ckpt", "david_mack-000016.ckpt", "alan_lee-000017.ckpt", "ed_mell-000017.ckpt", "michael_garmash-000017.ckpt"]

# Calculates the degree of difference between two _flattened_ models
def compare(model_a_weights, model_b_weights):
    diff = numpy.abs(model_a_weights - model_b_weights)
    return numpy.average(diff)

def process(model_paths):
    model_dicts = []
    for i in tqdm(model_paths, "Loading models from disk..."):
        model_dicts.append(torch.load(i, map_location="cpu"))

    flattened_models = []
    for i in tqdm(model_dicts, "Flattening models..."):
        tensor_list, _ = tensor_manager.get_tensors(i)
        flattened_models.append(flat.model_flatten(tensor_list)[0])
    
    # Find every unique pairing of indeces
    print("Finding model combinations...")
    model_combos = [comb for i in range(len(model_dicts) + 1) for comb in combinations(range(len(model_dicts)), i) if len(comb) == 2]
    combo_difference_scores = []
    merged_models = []
    merged_model_names = []

    for i in range(len(model_combos)):
        ind_a, ind_b = model_combos[i]
        model_a, model_b = model_dicts[ind_a], model_dicts[ind_b]

        name_a, name_b = model_paths[ind_a], model_paths[ind_b]
        merged_model_names += name_a + name_b + "0.5"
        
        merged_model = merge.merge(model_a, model_b, names=name_a+" and "+name_b)
        merged_models.append(merged_model)
        merged_model_tensors, _ = tensor_manager.get_tensors(merged_model)
        print("Flattening merged model...", end="")
        flattened_merged_model, _, _, _ = flat.model_flatten(merged_model_tensors)
        print(" Done.")

        differences = numpy.array([])
        for j in tqdm(range(len(flattened_models)), "Comparing to other models..."):
            differences = numpy.append(differences, compare(flattened_models[j], flattened_merged_model))
        av_difference = numpy.average(differences)
        combo_difference_scores.append(av_difference)

    sorted_models_and_scores = sorted(zip(combo_difference_scores, merged_models, merged_model_names))
    combo_difference_scores = [i[0] for i in  sorted_models_and_scores]
    merged_models = [i[1] for i in  sorted_models_and_scores]
    merged_model_names = [i[2] for i in  sorted_models_and_scores]

    print("Difference score results:")
    for i in range(len(merged_models)):
        print("[" + str(i+1) + "]: " + merged_model_names[i] + " - " + str(combo_difference_scores[i].item()))