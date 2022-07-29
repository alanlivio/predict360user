import pathlib
import pickle
from os.path import exists


class Savable():

    def save(self):
        with open(self._pickle_file(), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def _pickle_file(cls):
        return pathlib.Path(__file__).parent.parent / f'output/{cls.__name__}.pickle'

    @classmethod
    def load_or_create(cls):
        if exists(cls._pickle_file()):
            with open(cls._pickle_file(), 'rb') as f:
                print(f"loading {cls.__name__} from {cls._pickle_file()}")
                return pickle.load(f)
        else:
            return cls()
