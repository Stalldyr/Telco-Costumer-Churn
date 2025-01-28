import pandas as pd
import json

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import SVC
import shap

from datetime import datetime
from PlotBuilder import PlotBuilder

class Data():
    #╔════════════════════════════════════════════════════════════════════╗
    #║                         INITIALIZATION                             ║
    #╚════════════════════════════════════════════════════════════════════╝
    def __init__(self, path, save_log = False, save_to_file=False):
        self.df = pd.read_csv(path)
        self.categories = pd.DataFrame()
        self.document = ''
        self.save_log = save_log
        self.save_to_file = save_to_file
        self.txt_output = ['----------------------------------------------------------',
                           'LOG ' + datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                           '----------------------------------------------------------\n']

        with open('column_types.json') as file:
            self.column_types = json.load(file)

    def categorize_features(self): 
        '''Categorizes the features into key, numeric, categorical, ordinal, and boolean features'''

        #Renames some of the columns for easier use
        self.df = self.df.rename(columns={'gender': 'Male', 'tenure':'Tenure'})
        
        #Categorical values that could be mapped to boolean values
        self.boolean_mapping = {'Yes': True, 'No': False, 'No internet service': False,  'No phone service': False, 0: False, 1:True, 'Male': True, 'Female': False, 'DSL':True, 'Fiber optic': True, 'No': False}
        
        #Categorize the features. The features can be sorted into key, numeric, categorical, and boolean features. Boolean can be further divided into service and demographic features.
        self.column_types = {
            'key': ['customerID'], #Unique ID, used to identify the costumer
            'numerical': ['Tenure', 'TotalCharges', 'MonthlyCharges'], #Features with numerical values (int/float)
            'categorical': ['PaymentMethod', 'Contract'], 
            'ordinal': [],
            'boolean': ['SeniorCitizen', 'PaperlessBilling', 'Dependents', 'Churn', 'MultipleLines', 'OnlineSecurity', 'Male', 'OnlineBackup', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'DeviceProtection', 'InternetService', 'PhoneService', 'Partner'],
            'demographic': ['SeniorCitizen', 'PaperlessBilling', 'Dependents', 'Churn', 'Male', 'Partner'],
            'service': ['OnlineBackup', 'OnlineSecurity', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'DeviceProtection', 'InternetService', 'PhoneService', 'MultipleLines']
        }

    def data_processing_and_feature_engineering(self):
        #Cleans, categorizes, processes, and engineers the data for machine learning, and saves to a new file for quicker retrival
        self.categorize_features()
        self.data_cleaning()
        self.feature_engineering()
        self.df.to_csv('Telco-Customer-Churn-new.csv', index=False)
        with open('column_types.json', 'w') as outfile: 
            json.dump(self.column_types,outfile)

    #╔════════════════════════════════════════════════════════════════════╗
    #║                EXPLORATION, CLEANING, & ENGINEERING                ║
    #╚════════════════════════════════════════════════════════════════════╝
    def data_exploration(self):
        '''Performs exploratory data analysis'''

        pd.set_option('display.max_columns', None)
        
        self.txt_output.append('DATA EXPLORATION\n')
        self.txt_output.append('Number of features: ' + str(self.df.shape[1]))
        self.txt_output.append('Number of entries: ' + str(self.df.shape[0]) + "\n")
        
        self.txt_output.append("Unique values for each feature:")
        for col in self.df:
            self.txt_output.append(col + f'({self.df[col].nunique()}): ' + str(self.df[col].unique()))

        self.txt_output.append("\nDatatype for each feature:\n" + str(self.df.dtypes))

        self.txt_output.append("\nRatios for boolean features:")
        for col in self.df[self.column_types["boolean"]]:
            self.txt_output.append(f'{col}: ' + str(round(self.df[col].mean(),2)))
        
        self.txt_output.append("\nMetrics of numerical features:\n" + str(self.df.describe()) + "\n")
        
        self.txt_output.append("\nThe five first entries of each feature:")
        
        self.txt_output.append(str(self.df.head()))
        pd.reset_option('display.max_columns')
            

    def data_cleaning(self, cutoff_percentage = 5, write_to_csv = False):
        '''Cleans and orders the data for further use'''

        self.df['TotalCharges'] = self.df['TotalCharges'].replace(' ', '0').astype(float) #Fixes an issue where fresh costumer has an empty string rather than 0 as total cost
        cutoff = cutoff_percentage * self.rows // 100
        if self.df.isnull().sum().sum() > cutoff:
            print('Missing values exceed cutoff!')
        self.boolean_encoding(self.boolean_mapping)
        
        if write_to_csv:
            self.save_processed_data()

    def feature_engineering(self, write_to_csv = False):
        '''Creates new features based on excisting features'''

        self.df['NumServices'] = self.df[self.column_types["service"]].sum(axis=1)
        self.df['AverageCharges'] = self.df['TotalCharges']/(self.df['Tenure']+1) #Average charge per customer
        tenure_bucket_labels = ['<1 year', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5+ years']
        self.df['TenureBucket'] = pd.cut(
            self.df['Tenure'], bins=[-1, 12, 24, 36, 48, 60, float('inf')],
            labels=tenure_bucket_labels)
        self.df['Tenure_Contract'] = self.df['TenureBucket'].astype(str) + self.df['Contract']
        self.df['HighPaying'] = ((self.df['MonthlyCharges'] > self.df['MonthlyCharges'].median()).astype(int)).astype(bool)
        self.df['CostPerService'] = self.df['AverageCharges'] / (self.df['NumServices'] + 1)

        #Updates the feature category sets with new features
        self.column_types["numerical"].append('NumServices')
        self.column_types["numerical"].append('AverageCharges')
        self.column_types["categorical"].append('TenureContract')
        self.column_types["ordinal"].append({'TenureBucket': tenure_bucket_labels})
        #Highpaying-placeholder
        #CostPerService-placeholder

        if write_to_csv:
            self.save_processed_data()


    def feature_exploration(self, plot = False):
        '''Explores the data set in depth, analysing for patterns and insights.'''

        tenure_by_churn = self.df.groupby("Churn")["Tenure"].mean().reset_index()
        churn_by_contract = self.df.groupby("Contract")["Churn"].mean().reset_index()
        churn_by_payment_method = self.df.groupby("PaymentMethod")["Churn"].mean().reset_index()
        high_paying_churn = self.df.groupby("HighPaying")["Churn"].mean().reset_index()
        corr_matrix = self.df[["AverageCharges","Tenure", "NumServices", "Churn"]].corr()
        churn_by_payment_contract = self.df.groupby(["PaymentMethod", "Contract"])["Churn"].mean().unstack()
    

        self.txt_output.append('\nFEATURE EXPLORATION\n')
        self.txt_output.append("\nAverage tenure of churned and non-churned costumers:")
        self.txt_output.append(str(tenure_by_churn))
        self.txt_output.append("\nChurn by contract:")
        self.txt_output.append(str(churn_by_contract))
        self.txt_output.append("\nChurn by payment method:")
        self.txt_output.append(str(churn_by_payment_method))
        self.txt_output.append("\nChurn by high paying customers:")
        self.txt_output.append(str(high_paying_churn))
        self.txt_output.append("\nCorrelation Matrix:")
        self.txt_output.append(str(corr_matrix))
        self.txt_output.append("\nChurn by Payment Method and Contract:")
        self.txt_output.append(str(churn_by_payment_contract))

        if plot:
            featureplots = [
                PlotBuilder().set_title("Average tenure of churned and non-churned costumers").set_rotation(40).set_x_label(" ").set_tick_labels(["Churned", "Non-churned"]).create_barplot(tenure_by_churn,"Churn","Tenure"),
                PlotBuilder().set_title("Churn rate by contract").set_rotation(40).set_limits(y_lim=(0,1)).create_barplot(churn_by_contract,"Contract","Churn"),
                PlotBuilder().set_title("Churn rate by payment method").set_rotation(40).set_limits(y_lim=(0,1)).create_barplot(churn_by_payment_method,"PaymentMethod","Churn"),
                PlotBuilder().set_title("Churn rate by high paying customers").set_rotation(40).set_limits(y_lim=(0,1)).create_barplot(high_paying_churn,"HighPaying","Churn"),
                PlotBuilder().set_title("Correlation Matrix").create_heatmap(corr_matrix),
                PlotBuilder().set_title("Churn rate by payment method and contract").create_heatmap(churn_by_payment_contract)
            ]

            for plot in featureplots:
                self.save_plot(plot)


    def numerical_data_exploration(self,cols):
        for col in cols:
            print(self.df[col].describe(), '\n')
   
    #╔════════════════════════════════════════════════════════════════════╗
    #║                     MACHINE LEARNING MODELS                        ║
    #╚════════════════════════════════════════════════════════════════════╝
    def training_initialization(self,features,target):
        '''Initiates model training based on selected features and target'''

        X = self.df[features]
        y = self.df[target]

        self.txt_output.append('\nMODEL TRAINING\n')
        
        self.txt_output.append('FEATURES: ' + ', '.join(features))
        self.txt_output.append('TARGET: ' + target)

        self.txt_output.append('\nMODEL: RANDOM FOREST')
        #y_true, y_pred = self.model_training(X,y,MinMaxScaler(),RandomForestClassifier())        
        #self.model_score(y_true, y_pred)

        self.txt_output.append('\nMODEL: LOGISTIC REGRESSION')
        y_true, y_pred = self.model_training(X,y,MinMaxScaler(),LogisticRegression())
        self.model_score(y_true, y_pred)


    
    def model_training(self,X,y,scaler,model,random_state=42):
        '''Performs training and testing on the selected features, target and model'''

        X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=random_state)

        transformers=[
                ('num', scaler, [col for col in X_train.columns if col in self.column_types['numerical']]),
                ('cat', OneHotEncoder(handle_unknown='ignore'), [col for col in X_train.columns if col in self.column_types['categorical']])    
            ]
        
        ordinal_columns = [col for col in self.df.columns if col in self.column_types["ordinal"][0]]
        if ordinal_columns:
            transformers.append(('ord', OrdinalEncoder(categories=[self.column_types["ordinal"][0][order] for order in ordinal_columns]), ordinal_columns))


        preprocessor = ColumnTransformer(transformers=transformers)

        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
       
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)

        #self.shap_plots(X_train, X_test, model_pipeline)

        return y_test, y_pred
    
    def cross_validation(self,X,y,random_state=42):
            num_folds = 5
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
            svm_classifier = SVC(kernel='linear')

            cross_val_results = cross_val_score(svm_classifier, X, y, cv=kf)

            self.txt_output.append('Cross-validation results: ' + str(cross_val_results))
            self.txt_output.append('Mean cross-validation score: ' + str(round(cross_val_results.mean(),2)))

    def model_score(self,y_true,y_pred):
        '''Calculates scores for the model'''

        accuracy = round(accuracy_score(y_true,y_pred),2)
        precision = round(precision_score(y_true,y_pred),2)
        recall = round(recall_score(y_true,y_pred),2)
        f1 = round(f1_score(y_true,y_pred),2)

        self.txt_output.append('Accuracy: ' + str(accuracy))
        self.txt_output.append('Precision: ' + str(precision))
        self.txt_output.append('Recall: ' + str(recall))
        self.txt_output.append('F1-score: ' + str(f1))

    #╔════════════════════════════════════════════════════════════════════╗
    #║                        DATA VISUALIZATION                          ║
    #╚════════════════════════════════════════════════════════════════════╝    
    def shap_plots(self, X_train, X_test, model_pipeline):
        # Extract preprocessed data
        preprocessor = model_pipeline.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()
        
        X_train_preprocessed = pd.DataFrame(
            preprocessor.transform(X_train),
            columns=feature_names,
            index=X_train.index
        )

        X_test_preprocessed = pd.DataFrame(
            preprocessor.transform(X_test),
            columns=feature_names,
            index=X_test.index
        )

        #Create SHAP explainer
        explainer = shap.KernelExplainer(model_pipeline.predict, X_train_preprocessed)
        shap_values = explainer.shap_values(X_test_preprocessed)

        # Generate SHAP plots
        shap.summary_plot(shap_values, X_test_preprocessed)
        shap.dependence_plot('Tenure', shap_values, X_test_preprocessed)

    #╔════════════════════════════════════════════════════════════════════╗
    #║                   FILE PROCESSING AND LOGGING                      ║
    #╚════════════════════════════════════════════════════════════════════╝
    def read_file(self,path):
        self.log = open(path,'a')
    
    def save_processed_data(self, output_path):
        self.df.to_csv(output_path, index=False)

    def save_plot(self,plotbuild):
        plt.savefig(f'plots\{plotbuild.title.replace(" ","_")}_{plotbuild.type}.png',bbox_inches="tight")

    #╔════════════════════════════════════════════════════════════════════╗
    #║                         HELPER FUNCTIONS                           ║
    #╚════════════════════════════════════════════════════════════════════╝
    def boolean_encoding(self, boolean_mapping):
        for col in self.column_types['boolean']:
            self.df[col] = self.df[col].map(boolean_mapping)

    def _round_value(self, value, decimals=2):
        #Helper method to round numeric values consistently.
        if isinstance(value, (int, float)):
            return round(value, decimals)
        elif isinstance(value, pd.Series):
            return value.round(decimals)
        return value

        
def main(path,save_log=False,save_to_file = False):
    '''Main function. Initiates everything'''

    #Initializes data from path:
    telco = Data(path,save_log,save_to_file)

    features = ['NumServices', 'AverageCharges'] 
    target = 'Churn'

    #telco.cross_validation(X,y)

    telco.feature_exploration(plot=True)
    #telco.training_initialization(features,target)

    print('\n'.join(telco.txt_output))

    #Adds insights to a logfile
    if save_log:
        telco.read_file('Customer_Churn_log.txt')
        telco.log.write("\n".join(telco.txt_output) + '\n\n')
        telco.log.close()
        

if __name__ == '__main__':
    main('Telco-Customer-Churn-new.csv',save_log = True)