import importlib

class LazyLoader:
    """This class is used to lazily load a class from a string path."""
    def __init__(self, class_path, *args, **kwargs):
        self.class_path = class_path
        self.args = args
        self.kwargs = kwargs

    def __getstate__(self):
        return {"class_path": self.class_path, "args": self.args, "kwargs": self.kwargs}
    
    def __setstate__(self, state):
        self.class_path = state["class_path"]
        self.args = state["args"]
        self.kwargs = state["kwargs"]
    
    def __call__(self, **kwargs):
        local_config = {**self.kwargs, **kwargs}
        module_name, class_name = self.class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        model_instance = model_class(*self.args, **local_config)
        return model_instance