import os
from git import Repo
from os import path
from config import config
import re
import urllib.parse

models_path = config['models_path']

def load_model_repo(repo_url, local_path="models", force=False):
    # Extract the repository name from the URl
    parsed_url = urllib.parse.urlparse(repo_url)

    # Extract the last part of the path (repository name)
    path_parts = parsed_url.path.strip('/').split('/')
    repo_name = '/'.join(path_parts[-1:])
    
    # Construct the local path based on the base path and repository name
    local_path = path.abspath(local_path)
    local_path = path.join(local_path, repo_name)
    
    if not path.exists(local_path) or force:
        try:
            # Clone the Git repository to the specified local path
            print("Cloaning model repository from {}.".format('repo_url'))
            repo = Repo.clone_from(repo_url, local_path)
            print("Repository cloned successfully at {}.".fromat('local_path'))
        except Exception as e:
            print(f"Error cloning repository: {str(e)}")
    
    return local_path


def make_iterable_old(func):
    """make function func resive iterable input and return list of output
    """
    def wrapper(inp, *args, **kwargs):
        if hasattr(inp, '__iter__'):
            out = []
            for o in inp:
                out.append(func(o, *args, **kwargs))
            
            return out
        else:
            return func(inp, *args, **kwargs)
    return wrapper
            
            
import concurrent.futures
def make_iterable(func):
    """Make the function 'func' accept iterable input and return a list of output using parallel processing.
    """
    def wrapper(inp, *args, **kwargs):
        if hasattr(inp, '__iter__') and not isinstance(inp, str):
            # Check if the input is an iterable

            # Create a ThreadPoolExecutor for parallel processing
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Use list comprehension and concurrent processing to apply 'func' to each element
                results = list(executor.map(lambda x: func(x, *args, **kwargs), inp))

            return results
        else:
            # If the input is not iterable, apply 'func' to it directly
            return func(inp, *args, **kwargs)

    return wrapper