"""Module containing functions for parsing the input files g.mkm and rm.mkm"""

def preprocess(input_file, ws=3):
    """Return the lines without comments
    Args:
        input_file(path): input file to be processed.
    Returns:
        new_lines(list): list of the important strings
    """
    x = open(input_file, "r")
    lines = x.readlines()
    new_lines = []
    total_length = len(lines)
    for string in lines:
        if "#" in string: # Comments are implemented like in Python!
            index = string.find("#")
            new_lines.append(string[:index].strip("\n"))
        else:
            new_lines.append(string.strip("\n"))
        new_lines[-1] = new_lines[-1].rstrip()
    for i in range(len(new_lines)):
        if new_lines[i] == "":
            continue
        else:
            index_first = i
            break
    new_lines = new_lines[index_first:]
    counter = 0
    ll = len(new_lines)
    for i in range(len(new_lines)):
        if new_lines[i] == '':
            counter += 1
        else:
            pass
        if counter > ws:
            index_last = i
            break
        else:
            index_last = ll
    new_lines = new_lines[:index_last]
    return new_lines

def get_NGR_NR(input_list):
    """
    Get number of global and elementary reactions from the 
    input list from rm.mkm.
    """
    NGR = 0
    for i in range(len(input_list)):
        if input_list[i] != '':
            NGR += 1
        else:
            break
    NR = len(input_list) - NGR - 3
    return NGR, NR
        
def stoich_matrix(list):
    pass

def classify_species(list):
    pass
