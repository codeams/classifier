
# Removes system hidden file
# names from the given array.
# Supports macOS X.10+
def remove_system_files(list):
    for element in list:
        if element == ".DS_Store":
            list.remove(element)
    return list
