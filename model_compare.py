from itertools import combinations
import numpy
from tqdm import tqdm
import flat
import tensor_manager
import merge
import torch
from pathlib import Path
import operator

MENU_STR = """
[1] Iterate (merge-compare-delete loop)
[2] Merge all models in pool
[3] Delete model
[4] Load model from disk
[5] Compare and delete
[6] Save all models in pool to disk
[Q] Quit
"""

INITIAL_PATHS=[]

def remove_filetype(filename):
    for i in range(len(filename)):
        if filename[i] == ".":
            break
    return filename[:i]

# Calculates the degree of difference between two _flattened_ models
def compare(model_a_weights, model_b_weights):
    diff = numpy.abs(model_a_weights - model_b_weights)
    return numpy.average(diff)

class Application:
    def __init__(self):
        self.models = []
        self.model_names = []
        self.flattened_models = []

    def menu(self):
        valid = False
        while not valid:
            choice = input(MENU_STR + "> ").lower()
            if choice not in [str(i) for i in range(1,6)] + ["q"]:
                print("Please enter a valid option")
            else:
                valid = True
        return choice

    def load_from_disk(self, paths):
        self.model_names.extend([remove_filetype(m) for m in paths])
        new_model_dicts, new_flattened_models = [], []
        for i in tqdm(paths, "Loading models from disk..."):
            new_model_dicts.append(torch.load(i, map_location="cpu"))

        for i in tqdm(new_model_dicts, "Flattening models..."):
            tensor_list, _ = tensor_manager.get_tensors(i)
            new_flattened_models.append(flat.model_flatten(tensor_list)[0])
        
        self.models.extend(new_model_dicts)
        self.flattened_models.extend(new_flattened_models)

    def calc_diff_score(self, flat_model):
        differences = numpy.array([])
        for j in tqdm(range(len(self.flattened_models)), "Comparing to other models..."):
            differences = numpy.append(differences, compare(self.flattened_models[j], flat_model))
        av_difference = numpy.average(differences)
        return av_difference

    def sort_by_score(self, scores, models, model_names, flattened_models):
        zipped = zip(scores, models, model_names, flattened_models)
        sorted_models_and_scores = sorted(zipped, reverse=True, key=operator.itemgetter(0))
        scores = [i[0] for i in  sorted_models_and_scores]
        models = [i[1] for i in  sorted_models_and_scores]
        model_names = [i[2] for i in  sorted_models_and_scores]
        flattened_models = [i[3] for i in sorted_models_and_scores]
        return scores, models, model_names, flattened_models

    def combine_names(self, a, b, ratio):
        ratio = str(ratio)
        ratio = ratio[0] + "_" + ratio[2:]
        return "(" + a + ")" + "-" + "(" + b + ") " + ratio

    def get_unique_pairings(self):
        model_combos = []
        for i in range(len(self.models) + 1):
            for comb in combinations(range(len(self.models)), i):
                if len(comb) == 2:
                    model_combos.append(comb)
        return list(set(model_combos))
    
    def get_ratio_list(self, ratios):
        ratio_list = []
        for i in range(ratios):
            ratio_list.append(round((i + 1) / (ratios + 1), 2))
        print(ratio_list)
        return ratio_list

    def merge_all_and_compare(self, ratios): 
        # Find every unique pairing of indeces
        print("Finding model combinations...") 
        ratio_list = self.get_ratio_list(ratios)
        model_combos = self.get_unique_pairings()
        combo_difference_scores = []
        merged_models = []
        merged_model_names = []
        flattened_merged_models = []

        for i in range(len(model_combos)):
            ind_a, ind_b = model_combos[i]
            model_a, model_b = self.models[ind_a], self.models[ind_b]
            name_a, name_b = self.model_names[ind_a], self.model_names[ind_b]
            for r in ratio_list:
                merged_model_names.append(self.combine_names(name_a, name_b, r))

                print("Merging", name_a, "and", name_b, "with a ratio of", str(r), "...", end="")
                merged_model = merge.merge(model_a, model_b, r)
                merged_models.append(merged_model)
                merged_model_tensors, _ = tensor_manager.get_tensors(merged_model)
                print("Done.")

                print("Flattening merged model...", end="")
                flattened_merged_model, _, _, _ = flat.model_flatten(merged_model_tensors)
                flattened_merged_models.append(flattened_merged_model)
                print(" Done.")

                combo_difference_scores.append(self.calc_diff_score(flattened_merged_model))

        combo_difference_scores, merged_models, merged_model_names, flattened_merged_models = self.sort_by_score(combo_difference_scores, merged_models, 
                                                                                                            merged_model_names, flattened_merged_models)
        print("Difference score results:")
        for i in range(len(merged_models)):
            print("[" + str(i+1) + "]: " + merged_model_names[i] + " - " + str(combo_difference_scores[i].item()))
        
        return merged_models, merged_model_names, flattened_merged_models

    def display_pool(self):
        if self.models == []:
            print("Current model pool is empty.")
        else:
            print("\nCurrent model pool is:")
            for i in self.model_names:
                print("\t" + i)

    def process(self, model_paths=[]):
        if model_paths:
            self.load_from_disk(model_paths)

        cont = True
        while cont:
            self.display_pool()
            
            choice = self.menu()

            if choice == "1":
                correct = False
                while not correct:
                    print()
                    iterations = max(1, int(input("Number of iterations: ")))
                    ratios = min(9, int(input("Number of merges per model pair (1 means a 0.5 merge, 2 means a 0.33 and 0.66, etc.): ")))
                    to_keep = min(1, int(input("Number of models to keep after each iteration: ")))
                    print("Iterations:", str(iterations), ", Ratios: ", str(ratios), ", Models to keep: ", str(to_keep))
                    inp = input("Is this correct [y/n]: ").lower()
                    if inp == "y" or inp == "yes":
                        correct = True

                for i in range(iterations):
                    print("Iteration " + str(i + 1))
                    merged_models, merged_model_names, flattened_merged_models = self.merge_all_and_compare(ratios)
                    print("Adding first " + str(to_keep) + " to pool... ", end="")
                    self.models.extend(merged_models[:to_keep])
                    self.model_names.extend(merged_model_names[:to_keep])
                    self.flattened_models.extend(flattened_merged_models[:to_keep])
                    print("done.")
                
                print("Iterations complete.")

            elif choice == "2":
                ratios = min(9, int(input("Number of merges per model pair (1 means a 0.5 merge, 2 means a 0.33 and 0.66, etc.): ")))
                merged_models, merged_model_names, flattened_merged_models = self.merge_all_and_compare(ratios)
                
                to_keep = int(input("How many models would you like to keep? "))
                self.models.extend(merged_models[:to_keep])
                self.model_names.extend(merged_model_names[:to_keep])
                self.flattened_models.extend(flattened_merged_models[:to_keep])

                print("Added " + str(merged_model_names[:to_keep]) + " to model pool")

            elif choice == "3":
                print()
                for i in range(len(self.model_names)):
                    print("[" + str(i+1) + "] " + self.model_names[i])
                to_del = int(input("Choose a model to delete: "))
                to_del -= 1
                del self.model_names[to_del]
                del self.flattened_models[to_del]
                del self.models[to_del]
            
            elif choice == "4":
                name = input("\nPlease enter the name of the model you want to add: ")
                file = Path(name)
                if remove_filetype(name) in self.model_names:
                    print("That model is already in the pool")
                elif file.exists():
                    self.load_from_disk([file.name])
                else:
                    print("Sorry, that file doesn't seem to exist. Did you spell it correctly?")
            
            elif choice == "5":
                scores = []
                for m in tqdm(self.flattened_models, "Comparing models..."):
                    scores.append(self.calc_diff_score(m))
                scores, self.models, self.model_names, self.flattened_models = self.sort_by_score(scores, self.models, self.model_names, self.flattened_models)
                print()
                for i in range(len(self.model_names)):
                    print("["+str(i+1)+"] - " + self.model_names[i] + " - " + str(scores[i].item))
                to_keep = int(input("Enter number of models to _keep_ (from the top down): "))
                print("Deleting models " + str(to_keep+1) + " to " + str(len(self.models)))
                self.models = self.models[:to_keep]
                self.model_names = self.model_names[:to_keep]
                self.flattened_models = self.flattened_models[:to_keep] 
            
            elif choice == "6":
                confirmation = input("To confirm - you want to save ALL models in the pool to disk [y/n]").lower()
                if confirmation == "y" or confirmation == "yes":
                    for i in tqdm(range(len(self.models)), "Saving models to disk..."):
                        fname = self.model_names[i] + ".ckpt"
                        if Path(fname).exists():
                            print("A file called '" + fname + "' already exists. Skipping.")
                        else:
                            torch.save(self.models[i], open(fname, "wb"))
            
            elif choice == "q":
                sure = input("Are you sure? Models in the pool that aren't saved to disk will be lost. [y/n] ").lower()
                if sure == "y" or sure == "yes":
                    cont = False