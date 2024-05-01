import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(3872, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from sparktorch import SparkTorch, serialize_torch_obj
from pyspark.sql.functions import rand
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline, PipelineModel
from sparktorch import PysparkPipelineWrapper

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("examples") \
        .master('local[4]').config('spark.driver.memory', '4g') \
        .getOrCreate()

    # Read in mnist_train.csv dataset
    df = spark.read.option("inferSchema", "true").csv('~/mnist_train.csv').orderBy(rand()).repartition(2)

    network = Net()

    # Build the pytorch object
    torch_obj = serialize_torch_obj(
        model=network,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        lr=0.001
    )

    # Setup features
    vector_assembler = VectorAssembler(inputCols=df.columns[1:785], outputCol='features')

    # Demonstration of some options. Not all are required
    # Note: This uses the barrier execution mode, which is sensitive to the number of partitions
    spark_model = SparkTorch(
        inputCol='features',
        labelCol='_c0',
        predictionCol='predictions',
        torchObj=torch_obj,
        iters=50,
        verbose=1,
        validationPct=0.2,
        miniBatch=128
    )

    # Create and save the Pipeline
    p = Pipeline(stages=[vector_assembler, spark_model]).fit(df)
    p.write().overwrite().save('simple_cnn')

    # Example of loading the pipeline
    loaded_pipeline = PysparkPipelineWrapper.unwrap(PipelineModel.load('simple_cnn'))

    # Run predictions and evaluation
    predictions = loaded_pipeline.transform(df).persist()

    evaluator = MulticlassClassificationEvaluator(
        labelCol="_c0", predictionCol="predictions", metricName="accuracy")

    accuracy = evaluator.evaluate(predictions)
    print("Train accuracy = %g" % accuracy)
