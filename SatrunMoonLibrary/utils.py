import os

# General file structure debuggin utils
def list_all_folders(directory):
    """
    List all folders in the given directory.

    Parameters:
        directory (str): The path of the directory to list folders from.

    Returns:
        list: A list of folder names.
    """
    # Get a list of all entries in the directory
    entries = os.listdir(directory)
    
    # Filter out only directories
    folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
    
    return folders

def list_all_files(directory):
    """
    List all files in the given directory.

    Parameters:
        directory (str): The path of the directory to list files from.

    Returns:
        None
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(os.path.join(root, file))