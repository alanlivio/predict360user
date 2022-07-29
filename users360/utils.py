import pathlib
import pickle
from os.path import exists

class SavableSingleton:
    _instance = None

    @classmethod
    def pickle_name(cls):
        return pathlib.Path(__file__).parent.parent / f'output/{cls.__name__}.pickle'
        
    @classmethod
    def save(cls):
        with open(cls.pickle_name(), 'wb') as f:
            pickle.dump(cls._instance, f)

    @classmethod
    def singleton(cls):
        if cls._instance is None:
            if exists(cls.pickle_name()):
                with open(cls.pickle_name(), 'rb') as f:
                    print(f"loading {cls.__name__}. from {cls.pickle_name()}")
                    cls._instance = pickle.load(f)
            else:
                cls._instance = cls()
        return cls._instance