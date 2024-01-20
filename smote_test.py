##Set Up a PySpark Testing Environment

import unittest
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

class PreSmoteDfProcessTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local[2]").appName("PreSmoteDfProcessTest").getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()


##Test 1: Correct Input
##This test checks if the function behaves as expected when given correct input:

def test_pre_smote_df_process_correct_input(self):
    # Arrange
    data = [(1, "a", 0.5), (2, "b", 0.3), (3, "a", 0.8), (4, "b", 0.2)]
    df = self.spark.createDataFrame(data, ["id", "category", "value"])
    num_cols = ["value"]
    cat_cols = ["category"]
    target_col = "id"
    
    # Act
    result_df = pre_smote_df_process(df, num_cols, cat_cols, target_col)
    
    # Assert
    expected_columns = {"features", "category_index", "label"}
    actual_columns = set(result_df.columns)
    self.assertSetEqual(actual_columns, expected_columns, "The DataFrame should have features, category_index, and label columns.")
    self.assertEqual(result_df.select("category_index").distinct().count(), 2, "There should be two distinct values in category_index.")


##Test 2: Incorrect Target Classes
##This test ensures the function raises a ValueError when the target column doesn't have exactly two classes:

def test_pre_smote_df_process_incorrect_target_classes(self):
    # Arrange
    data = [(1, "a", 0.5), (2, "b", 0.3), (3, "a", 0.8), (4, "b", 0.2), (5, "b", 0.1)]
    df = self.spark.createDataFrame(data, ["id", "category", "value"])
    num_cols = ["value"]
    cat_cols = ["category"]
    target_col = "id"
    
    # Act & Assert
    with self.assertRaises(ValueError):
        _ = pre_smote_df_process(df, num_cols, cat_cols, target_col)



