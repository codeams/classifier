
def remove_system_files(file_paths):
    for file_path in file_paths:
        if file_path == ".DS_Store":
            file_paths.remove(file_path)
    return file_paths
