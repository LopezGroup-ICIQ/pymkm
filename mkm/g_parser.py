"""Functions for parsing g.mkm input file (system energetics"""

def preprocess_g(input_file):
    """
    Preprocess the plain text rm.mkm input file, removing 
    comments, blank lines and trailing white spaces.
    Args:
        input_file(path): input file to be processed.
        ws(int): number of blank lines between global and elementary reactions. 
                 Default to 3.
    Returns:
        new_lines(list): list of the important strings
    """
    x = open(input_file, "r")
    lines = x.readlines()
    new_lines = []
    total_length = len(lines)
    for string in lines:
        if "#" in string:
            index = string.find("#")
            new_lines.append(string[:index].strip("\n"))  # Remove comments 
        else:
            new_lines.append(string.strip("\n"))
        new_lines[-1] = new_lines[-1].rstrip()   # Remove trailing white spaces
    for i in range(len(new_lines)):
        if new_lines[i] == "":
            continue
        else:
            index_first = i   # index of first line after blank lines
            break
    new_lines = new_lines[index_first:]  # Remove empty lines at the beginning
    counter = 0
    index_last = len(new_lines)
    for i in range(index_last):
        if new_lines[i] == '':
            counter += 1
        if counter > 6:  # 3 blank lines between global and elementary reactions
            index_last = i  # index of first blank line at the end
            break
    new_lines = new_lines[:index_last]  # Remove trailing blank lines
    return new_lines