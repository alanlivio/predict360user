from sklearn.model_selection import train_test_split

from .data import *


def get_train_test_split(train_entropy, test_entropy, perc_test) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = Data.instance().df_trajects
    if (train_entropy != 'all') and (test_entropy != 'all'):
        assert not df['traject_entropy_class'].empty, "dataset has no 'traject_entropy_class' collumn, run -calc_trajects_entropy"
        X_train, X_test = train_test_split(df, test_size=perc_test, random_state=1)
    else:
        if train_entropy == 'all':
            X_train, _ = train_test_split(df, test_size=perc_test, random_state=1)
        else:
            X_train, _ = train_test_split(df[df['traject_entropy_class'] == train_entropy], test_size=perc_test, random_state=1)
        if test_entropy == 'all':
            _, X_test = train_test_split(df, test_size=perc_test, random_state=1)
        else:
            _, X_test = train_test_split(df[df['traject_entropy_class'] == test_entropy], test_size=perc_test, random_state=1)
    return X_train, X_test
