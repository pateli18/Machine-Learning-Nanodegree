# Machine Learning Engineer Nanodegree
## Capstone Proposal
Ihsaan Patel  
June 5th, 2017

## Proposal

### Domain Background
Unplanned hospital readmissions, which occur when a patient is admitted into a hospital within a relatively short time period (e.g. 30,60,90 days) post discharge from a hospital, are a burden not only to patients but the healthcare system as well. One study examining Medicare readmissions estimated that unplanned readmissions represent a $12 billion burden on the U.S. healthcare system.<sup>1</sup> While interventions that could reduce readmission rates post discharge exist,<sup>2</sup> ensuring cost effective use of these interventions requires the ability to actually predict which patients are most at-risk for readmission. Studies attempting to predict readmission rates using machine learning have been done previously,<sup>3</sup> however advances in both natural language processing and machine learning algorithms since the time of those studies could lead to meaningful improvements in predictive capabilities.
 
### Problem Statement
Given both administrative and clinical hospital data, is it possible to accurately predict a patient’s risk of readmission at the time of discharge from the hospital? Readmission is defined as admittance into the same or another hospital facility within a defined period of time (30, 60, and 90 day time periods will be modeled out). The problem is therefore a standard classification one (a patient was either readmitted within the defined time period or they were not) well-suited to machine learning techniques.
 
### Datasets and Inputs
The Medical Information Mart for Intensive Care (MIMIC) III database will be the source of data for this problem. MIMIC-III is a “large, freely-available database comprising deidentified health-related data associated with over forty thousand patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012.” It includes a number of datasets containing demographic information, vital signs taken every hour, while the patient was in the hospital, laboratory test results, procedures, medications, caregiver notes, imaging reports and mortality.<sup>4</sup> The dataset itself contains 40,000+ patients and 10,000+ instances of readmission, which should allow for the development of strong machine learning models.
 
Due to the nature of US regulations around patient data confidentiality and the anonymized nature of the MIMIC-III database, bringing in outside datasets is a non-starter. Therefore, the datasets in MIMIC-III will likely be the only ones used in this project, however the large number of both datasets and records within the database should still allow for effective training and testing of machine learning models.
 
### Solution Statement
There are a number of machine learning classifiers that could be effective in helping to solve the prediction problem, including logistic regression, support vector machines, random forests, and gradient boosted trees. Text feature engineering for these models will be done using tf-idf and potentially some unsupervised learning techniques to reduce the dimensionality of the text features, including PCA and K-means. Each of the classifiers’ hyperparameters will be tuned using k-fold cross-validation, and ensembles of these classifiers will also be tested, with the complexity of these ensembles weighed against any improvements in the chosen performance metrics. 
 
### Benchmark Models
Two models will be used as benchmarks for the project: the LACE index and a logistic regression model with L1 regularization (LASSO). The LACE (length of stay, acuity of admission, Charlson comorbidity index, CCI, and number of emergency department visits in preceding 6 months) index is a simple model used in the medical community to predict risk of readmission, although its actual effectiveness has been questioned.<sup>5</sup> While outperforming the index should be straightforward, it may set the benchmark too low given its aforementioned ineffectiveness, and so the project models will also be evaluated against the logistic LASSO regression model as this was the main machine learning model used in a previous study on the issue.<sup>6</sup>
 
### Evaluation Metrics
To compare the performance of the project’s machine learning models with previous work done on the problem, the primary evaluation metrics will be area under the ROC curve (LACE scores will be min-max normalized to evaluate them based on this metric) and cost effectiveness. Area under the ROC curve should provide a useful comparison on the trade-off between true and false positives, which represent costly mistakes if healthcare providers end up spending money on readmission prevention interventions that are not needed. Cost effectiveness will be measured based on a previous study,<sup>7</sup> which calculated both a per patient cost for an unplanned readmission as well as for a readmission prevention program. The total cost of the model would be calculated as the sum of the following for each patient:
 
Cost<sub>Intervention</sub> + Probability<sub>ReadmissionAllPatients</sub> x (1 - Reduction<sub>ReadmissionRisk</sub>) x Cost<sub>Readmission</sub>    

The cost effectiveness of the model would be its total cost relative to the cost of a program without any interventions.
 
### Project Design
The workflow for the project should go as follows:
 
1. **Clean Data and Generate Classes for Each Patient**: Some patients may have multiple readmissions, others might have died during their hospital stay, and so the dataset should be cleaned for only those records that are applicable to the problem
2. **Exploratory Data Analyses**: Due to the large number of potential features, some EDA is needed to both understand the features and get a better sense of which ones might make useful predictors
3. **Feature Engineering**: Create features from text fields like physician notes using tf-idf and unsupervised learning as well as other features that may not be able to be plugged directly into the model
4. **Feature Selection**: Train and cross-validate models to figure out which features to include
5. **Hyper-Parameter Tuning**: Tune hyper-parameters for the models using k-fold cross-validation
6. **Evaluate Models**: Evaluate models based on aforementioned performance metrics
7. **Create and Evaluate Ensemble Models**: Compare the performance of ensemble models to individual models and evaluate complexity-performance trade-off

-----------
1) http://www.nejm.org/doi/pdf/10.1056/NEJMsa0803563
2) http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0109264
3) http://content.healthaffairs.org/content/33/7/1123.abstract
4) http://content.healthaffairs.org/content/33/7/1123.abstract
5) https://www.hindawi.com/journals/bmri/2015/169870/
6) http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0109264
7) http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0109264
