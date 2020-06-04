import context
from modules.data_preprocess import DataCleaning, DataQualityCheck
from modules.db_connect import dbConnect
from test_sequence import sale

qc = DataQualityCheck()
result = qc.generate_QC(sale)
print(result)