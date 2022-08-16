from users360 import *

if __name__ == "__main__":
    Data.singleton().load_dataset()
    df = Data.singleton().df_trajects
    print(f"df_trajects.size={df.size}")
    calc_trajects_entropy()
    Data.singleton().save()
