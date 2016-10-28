# Loan-Fraud-Detection-with-Spark
This project aims to determine the result (approved or declined) for a loan application based on  the information of the applicant.

# Dataset

The data is from the German Credit Data Set which classifies people described by a set of attributes about credit risks. 

https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29

The data has been downloaded and saved as credit_data.txt.

# Package the Scala project with SBT

1. Download sbt.rar and unpack it into C:\project\

2. In the folder: C:\project\sbt\, run:
$ sbt assembly

3. Copy the executable JAR file from the folder C:\project\sbt\target\scala-2.11\ to the folder C:\project\, copy the data file credit_data.txt to the folder C:\project\

4. Run the Spark program:
$ spark-submit LoanFraudDetection-assembly-1.0.jar
