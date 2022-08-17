from users360 import *

if __name__ == "__main__":
    logging.info(f"load_dataset")
    Data.singleton().load_dataset()
    df = Data.singleton().df_trajects
    logging.info(f"df_trajects.size={df.size}")
    logging.info(f"calc_trajects_entropy")
    calc_trajects_entropy()
    Data.singleton().save()
