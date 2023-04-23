import model_compare

INITIAL_MODELS = [] # Add the filenames of models for them to be loaded immediately

if __name__ == "__main__":
    a = model_compare.Application()
    a.process(INITIAL_MODELS)