C:\Python27\python.exe feature.py smalltrain_data.tsv smallvalid_data.tsv smalltest_data.tsv dict.txt model1_formatted_train.tsv model1_formatted_valid.tsv model1_formatted_test.tsv 1

C:\Python27\python.exe feature.py smalltrain_data.tsv smallvalid_data.tsv smalltest_data.tsv dict.txt model2_formatted_train.tsv model2_formatted_valid.tsv model2_formatted_test.tsv 2


C:\Python27\python.exe lr.py model1_formatted_train.tsv model1_formatted_valid.tsv model1_formatted_test.tsv dict.txt model1train_out.labels model1test_out.labels model1metrics_out.txt 30

C:\Python27\python.exe lr.py model2_formatted_train.tsv model2_formatted_valid.tsv model2_formatted_test.tsv dict.txt model2train_out.labels model2test_out.labels model2metrics_out.txt 30 mod2_weights.txt


tar -cvf lr.tar feature.py lr.py




C:\Python27\python.exe feature.py train_data.tsv valid_data.tsv test_data.tsv dict.txt model1_formatted_train.tsv model1_formatted_valid.tsv model1_formatted_test.tsv 1

C:\Python27\python.exe feature.py train_data.tsv valid_data.tsv test_data.tsv dict.txt model2_formatted_train.tsv model2_formatted_valid.tsv model2_formatted_test.tsv 2


C:\Python27\python.exe lr.py model1_formatted_train.tsv model1_formatted_valid.tsv model1_formatted_test.tsv dict.txt model1train_out.labels model1test_out.labels model1metrics_out.txt 60 mod1_weights.txt

C:\Python27\python.exe lr.py model2_formatted_train.tsv model2_formatted_valid.tsv model2_formatted_test.tsv dict.txt model2train_out.labels model2test_out.labels model2metrics_out.txt 30 mod2_weights.txt
