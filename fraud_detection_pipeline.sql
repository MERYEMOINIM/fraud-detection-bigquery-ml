-- ============================================================================
-- FRAUD DETECTION ON FINANCIAL TRANSACTIONS WITH BIGQUERY ML
-- ----------------------------------------------------------------------------
-- Full pipeline: load -> explore -> feature engineering -> models -> predict
-- Based on Google Cloud lab GSP774. Dataset: PaySim (Kaggle).
-- Steps 1 to 3 run in Cloud Shell (bash). Steps 4+ run in the BigQuery SQL editor.
-- ============================================================================


-- ============================================================================
-- STEP 1 & 2 — LOAD THE DATA (run these in CLOUD SHELL, not the SQL editor)
-- ============================================================================
-- # Download and unzip the dataset from Google's public storage
-- gsutil cp gs://spls/gsp774/archive.zip .
-- unzip archive.zip
--
-- # Create shortcut variables to avoid retyping long names
-- export DATA_FILE=PS_20174392719_1491204439457_log.csv
-- export PROJECT_ID=<your_project_id>   # your real project id, no < > brackets
--
-- # Create a BigQuery dataset (a container for tables and models)
-- bq mk --dataset $PROJECT_ID:finance
--
-- # Create a Cloud Storage bucket and stage the CSV inside it
-- gsutil mb gs://$PROJECT_ID
-- gsutil cp $DATA_FILE gs://$PROJECT_ID
--
-- # Load the CSV into a BigQuery table.
-- # --autodetect reads column names and types automatically from the first row.
-- bq load --autodetect --source_format=CSV --max_bad_records=100000 \
--   finance.fraud_data gs://$PROJECT_ID/$DATA_FILE


-- ============================================================================
-- STEP 3 — EXPLORE THE DATA
-- ============================================================================

-- How many fraudulent transactions per transaction type?
-- GROUP BY groups rows by category; count(*) counts rows in each group.
SELECT type, isFraud, count(*) as cnt
FROM `finance.fraud_data`
GROUP BY isFraud, type
ORDER BY type;

-- Fraud proportion within the two relevant types (TRANSFER and CASH_OUT).
-- WHERE filters the rows before grouping.
SELECT isFraud, count(*) as cnt
FROM `finance.fraud_data`
WHERE type in ("CASH_OUT", "TRANSFER")
GROUP BY isFraud;

-- Top 10 largest transactions, to inspect extreme values.
SELECT *
FROM `finance.fraud_data`
ORDER BY amount desc
LIMIT 10;

-- KEY FINDING: fraud only appears in TRANSFER and CASH_OUT, and is < 1% of data.
-- The dataset is heavily imbalanced.


-- ============================================================================
-- STEP 4 — FEATURE ENGINEERING & DATA PREPARATION
-- ============================================================================
-- We create new features, filter irrelevant transaction types, and undersample
-- the non-fraud majority so the fraud signal becomes clearer to the models.

CREATE OR REPLACE TABLE finance.fraud_data_sample AS
SELECT
    type, amount, nameOrig, nameDest,
    oldbalanceOrg as oldbalanceOrig,   -- standardize the column name
    newbalanceOrig, oldbalanceDest, newbalanceDest,

    -- New engineered features that expose suspicious behaviour:
    if(oldbalanceOrg = 0.0, 1, 0)  as origzeroFlag,   -- origin account was empty
    if(newbalanceDest = 0.0, 1, 0) as destzeroFlag,   -- destination stays empty
    round((newbalanceDest - oldbalanceDest - amount)) as amountError, -- balance mismatch

    generate_uuid() as id,   -- unique id per transaction
    isFraud
FROM finance.fraud_data
WHERE
    type in ("CASH_OUT", "TRANSFER")           -- keep only fraud-prone types
    AND (isFraud = 1 OR (RAND() < 10/100));     -- keep ALL fraud + 10% of non-fraud

-- Hold out 20% as a TEST set, kept aside and never used for training.
CREATE OR REPLACE TABLE finance.fraud_data_test AS
SELECT * FROM finance.fraud_data_sample
WHERE RAND() < 20/100;

-- The remaining rows become the MODELLING set (~228k transactions).
-- EXCEPT DISTINCT removes the test rows so there is no overlap.
CREATE OR REPLACE TABLE finance.fraud_data_model AS
SELECT * FROM finance.fraud_data_sample
EXCEPT DISTINCT SELECT * FROM finance.fraud_data_test;

-- WHY UNDERSAMPLING? Fraud is so rare that a model can hit 99.9% accuracy by
-- always predicting "not fraud". Reducing the non-fraud majority forces the
-- model to actually learn the fraud pattern.


-- ============================================================================
-- STEP 5 — UNSUPERVISED MODEL: K-MEANS ANOMALY DETECTION
-- ============================================================================

-- Group transactions into 5 clusters based on similarity.
-- Note: isFraud is NOT used here -- that is what makes it "unsupervised".
CREATE OR REPLACE MODEL finance.model_unsupervised
OPTIONS(model_type='kmeans', num_clusters=5) AS
SELECT amount, oldbalanceOrig, newbalanceOrig, oldbalanceDest,
       newbalanceDest, type, origzeroFlag, destzeroFlag, amountError
FROM `finance.fraud_data_model`;

-- Profile fraud per cluster.
-- ML.PREDICT assigns each transaction to a cluster (centroid_id).
-- sum(isfraud) counts real frauds per cluster; count(*) is the total.
SELECT
  centroid_id, sum(isfraud) as fraud_cnt, count(*) total_cnt
FROM ML.PREDICT(MODEL `finance.model_unsupervised`,
     (SELECT * FROM `finance.fraud_data_test`))
GROUP BY centroid_id
ORDER BY centroid_id;

-- IDEA: without knowing which transactions are fraud, the model groups similar
-- ones together. Fraud tends to concentrate in small, anomalous clusters.
-- Cluster quality is measured by the Davies-Bouldin index (lower = better).


-- ============================================================================
-- STEP 6 — SUPERVISED MODEL 1: LOGISTIC REGRESSION
-- ============================================================================

-- Supervised: we give the model the answers via INPUT_LABEL_COLS = ["isfraud"].
CREATE OR REPLACE MODEL finance.model_supervised_initial
OPTIONS(model_type='LOGISTIC_REG', INPUT_LABEL_COLS = ["isfraud"]) AS
SELECT type, amount, oldbalanceOrig, newbalanceOrig,
       oldbalanceDest, newbalanceDest, isFraud
FROM finance.fraud_data_model;

-- Feature importance: which variables drive the prediction the most?
-- standardize=true removes the effect of scale so weights are comparable.
SELECT *
FROM ML.WEIGHTS(MODEL `finance.model_supervised_initial`,
     STRUCT(true AS standardize));

-- MOST IMPORTANT FEATURES: oldbalanceOrig and type.
-- METRICS THAT MATTER FOR FRAUD:
--   precision -> of flagged frauds, how many were truly fraud
--   recall    -> of all real frauds, how many were caught
--   ROC / AUC -> overall ability to separate fraud from genuine (>0.7 acceptable)
--   accuracy alone is MISLEADING on imbalanced data.


-- ============================================================================
-- STEP 7 — SUPERVISED MODEL 2: BOOSTED TREE
-- ============================================================================

-- A more powerful model: many decision trees combined (gradient boosting).
-- Same columns as logistic regression, so the two can be compared fairly.
-- Note: this model takes noticeably longer to train.
CREATE OR REPLACE MODEL finance.model_supervised_boosted_tree
OPTIONS(model_type='BOOSTED_TREE_CLASSIFIER', INPUT_LABEL_COLS = ["isfraud"]) AS
SELECT type, amount, oldbalanceOrig, newbalanceOrig,
       oldbalanceDest, newbalanceDest, isFraud
FROM finance.fraud_data_model;


-- ============================================================================
-- STEP 8 — EVALUATE & COMPARE THE MODELS
-- ============================================================================

-- ML.EVALUATE computes performance metrics for a model.
-- Store the logistic regression results first.
CREATE OR REPLACE TABLE finance.table_perf AS
SELECT "Initial_reg" as model_name, *
FROM ML.EVALUATE(MODEL `finance.model_supervised_initial`,
     (SELECT * FROM `finance.fraud_data_model`));

-- Append the boosted tree results to the same table for side-by-side comparison.
INSERT finance.table_perf
SELECT "boosted_tree" as model_name, *
FROM ML.EVALUATE(MODEL `finance.model_supervised_boosted_tree`,
     (SELECT * FROM `finance.fraud_data_model`));

-- CHAMPION MODEL: the boosted tree outperformed logistic regression.


-- ============================================================================
-- STEP 9 — PREDICT FRAUD ON HELD-OUT TEST DATA
-- ============================================================================

-- Apply the model to the untouched test set.
-- predicted_isfraud_probs is a nested array of probability scores;
-- unnest() expands it so we can filter on the fraud probability.
-- Keep transactions predicted as fraud (label = 1) with probability > 0.5.
SELECT id, label as predicted, isFraud as actual
FROM ML.PREDICT(MODEL `finance.model_supervised_initial`,
     (SELECT * FROM `finance.fraud_data_test`)),
     unnest(predicted_isfraud_probs) as p
WHERE p.label = 1 AND p.prob > 0.5;

-- RESULT: among transactions flagged as fraud, the actual fraud rate is far
-- higher than in the overall test set -- the model concentrates fraud into
-- its predictions successfully.
