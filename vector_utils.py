import importlib

# This will dynamically hold the backend module
backend = None  

def load_vector_utils(backend_name):
    """
    Load vector utility functions dynamically based on the backend name and assign it to the `backend` variable.
    
    Args:
        backend_name (str): Name of the backend ("annoy" or "qdrant").
    """
    global backend

    # Map backend name to module
    if backend_name == "annoy":
        module_name = "annoy_utils"
    elif backend_name == "qdrant":
        module_name = "qdrant_utils"
    else:
        raise ValueError(f"Unsupported backend: {backend_name}")

    # Dynamically import the module
    try:
        backend = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_name}'. Make sure it is installed and accessible.") from e
