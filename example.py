import os
os.getcwd()
import sagemaker
import boto3
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.session import s3_input, Session
bucket_name = "examplecaodata"
myregion=boto3.session.Session().region_name
print(myregion)
#!aws s3 mb s3://examplecaodata --region us-east-2
# create s3 bucket data folder
s3 = boto3.resource('s3')
try:
    if myregion=="us-east-2":
        s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': myregion})
    print("S3 bucket created!")
except Exception as e:
    print("error:",e)
    
# output path
prefix="xgboost-as-a-built-in-algo"
output_path=f"s3://{bucket_name}/{prefix}/output"
print(output_path)

import pandas as pd
import urllib
data_url = "https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv"
try:
    urllib.request.urlretrieve(data_url, "bank_clean.csv")
except Exception as e:
    print("data load error:",e)

model_data = pd.read_csv("./bank_clean.csv",index_col=0)
model_data.head()

# data split
from sklearn.model_selection import train_test_split
data = model_data.drop(['y_no', 'y_yes'], axis="columns")
labels = model_data['y_yes']
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.30, random_state=1729)

len(data_train),len(data_test)
pd.concat([labels_train,data_train], axis=1).to_csv('train.csv', index=False, header=False)
pd.concat([labels_test,data_test], axis=1).to_csv('test.csv', index=False, header=False)
# upload data into s3 buckets
# declare the data path

boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')

boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test.csv')).upload_file('test.csv')
s3_input_test = sagemaker.TrainingInput(s3_data='s3://{}/{}/test'.format(bucket_name, prefix), content_type='csv')
#aws s3 ls {bucket_name}/{prefix}/ --recursive


# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
# specify the repo_version depending on your preference.
container = get_image_uri(region_name = boto3.Session().region_name,
                          repo_name = 'xgboost', 
                          repo_version='latest')

hyperparameters = {
        "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "subsample":"0.8",
        "objective":"binary:logistic",
        "num_round":50
        }

# construct a SageMaker estimator that calls the xgboost-container
estimator = sagemaker.estimator.Estimator(image_uri=container, 
                                          hyperparameters=hyperparameters,
                                          role=sagemaker.get_execution_role(),
                                          train_instance_count=1, 
                                          instance_type='ml.m5.2xlarge', 
                                          train_volume_size=5, # 5 GB 
                                          output_path=output_path,
                                          train_use_spot_instances=True,
                                          train_max_run=300,
                                          train_max_wait=600)
estimator.fit({'train': s3_input_train,'validation': s3_input_test})
#! aws s3 ls {bucket_name}/{prefix}/output/ --recursive | grep model

xgb_predictor = estimator.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')

from sagemaker.serializers import CSVSerializer
import numpy as np
test_data_array = data_test.values #load the data into an array
xgb_predictor.content_type = 'text/csv' # set the data type for an inference
xgb_predictor.serializer = CSVSerializer() # set the serializer type

predictions = xgb_predictor.predict(test_data_array).decode('utf-8') # predict!
predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array
print(predictions_array.shape)

import sklearn

# confusion matrix
cutoff=0.5
print(sklearn.metrics.confusion_matrix(labels_test, np.where(predictions_array > cutoff, 1, 0)))
print(sklearn.metrics.classification_report(labels_test, np.where(predictions_array > cutoff, 1, 0)))


sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()