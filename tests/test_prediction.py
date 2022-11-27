import unittest

from users360.prediction import get_train_test_split


class Test(unittest.TestCase):

  def test_prediction_train_test_split(self) -> None:
    # x_train all
    x_train, x_test = get_train_test_split('all', 'all', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['low', 'medium', 'hight']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['low', 'medium', 'hight']
    
    x_train, x_test = get_train_test_split('all', 'hight', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['low', 'medium', 'hight']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == 'hight'
    
    x_train, x_test = get_train_test_split('all', 'medium', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['low', 'medium', 'hight']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['medium']
    
    x_train, x_test = get_train_test_split('all', 'low', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['low', 'medium', 'hight']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['low']
    
    # x_train low
    x_train, x_test = get_train_test_split('low', 'all', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['low']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['low', 'medium', 'hight']
    
    x_train, x_test = get_train_test_split('low', 'low', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['low']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['low']
    
    x_train, x_test = get_train_test_split('low', 'medium', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['low']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['medium']
    
    x_train, x_test = get_train_test_split('low', 'hight', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['low']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == 'hight'
    
    
    # x_train medium
    x_train, x_test = get_train_test_split('medium', 'all', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['medium']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['low', 'medium', 'hight']
    
    x_train, x_test = get_train_test_split('medium', 'low', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['medium']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['low']
    
    x_train, x_test = get_train_test_split('medium', 'medium', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['medium']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['medium']
    
    x_train, x_test = get_train_test_split('medium', 'hight', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == ['medium']
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == 'hight'
    
    # x_train hight
    x_train, x_test = get_train_test_split('hight', 'all', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == 'hight'
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['low', 'medium', 'hight']
    
    x_train, x_test = get_train_test_split('hight', 'low', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == 'hight'
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['low']
    
    x_train, x_test = get_train_test_split('hight', 'medium', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == 'hight'
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == ['medium']
    
    x_train, x_test = get_train_test_split('hight', 'hight', 0.2)
    unique_l = x_train['traject_entropy_class'].unique()
    assert unique_l == 'hight'
    unique_l = x_test['traject_entropy_class'].unique()
    assert unique_l == 'hight'
    
