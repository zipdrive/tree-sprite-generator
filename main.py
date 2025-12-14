import genus

def generate_samples(*args: genus.Genus):
    '''
    Deletes all files in the genus directory, and generates samples for the genus.
    '''
    import os 
    for k in range(len(args)):
        directory: str = f"file/{args[k].name}"
        if os.path.isdir(directory):
            for existing_file in os.listdir(directory):
                os.remove(directory + "/" + existing_file)
    for k in range(len(args)):
        for _ in range(5):
            args[k].generate_tree_structure(render_leaves=False, render_bare=False)

def generate_all_renders(*args: genus.Genus):
    '''
    Deletes all files in the genus directory, and generates renders for the genus.
    '''
    import os 
    for k in range(len(args)):
        directory: str = f"file/{args[k].name}"
        for existing_file in os.listdir(directory):
            os.remove(directory + "/" + existing_file)
    for k in range(len(args)):
        for _ in range(20):
            args[k].generate_tree_structure()

generate_samples(*genus.WILLOW_ALL)
print("Done.")