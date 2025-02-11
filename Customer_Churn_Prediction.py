import pandas as pd
import json

from datetime import datetime
from PlotBuilder import PlotBuilder
from ModelBuilder import ModelBuilder

#MACHINE LEARNING MODELS
RANDOM_FOREST = "random forest"
LOGISTIC_REGRESSION = "logistic regression"
XGBOOST = 'xgboost'

class Data():
    #╔════════════════════════════════════════════════════════════════════╗
    #║                         INITIALIZATION                             ║
    #╚════════════════════════════════════════════════════════════════════╝
    def __init__(self, path, save_log = False):
        with open('column_types.json') as file:
            self.column_types = json.load(file)
        self.df = pd.read_csv(path, dtype = {key: 'category' for key in self.column_types['categorical']+ list(self.column_types['ordinal'].keys())})
        self.categories = pd.DataFrame()

        self.save_log = save_log
        
        self.txt_output = ['----------------------------------------------------------',
                           'LOG ' + datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                           '----------------------------------------------------------\n']
        

    #╔════════════════════════════════════════════════════════════════════╗
    #║                   DATA CLEANING & ENGINEERING                      ║
    #╚════════════════════════════════════════════════════════════════════╝

    def categorize_features(self): 
        '''Categorizes the features into key, numeric, categorical, ordinal, and boolean features'''

        #Ordinal values that can be mapped to labels
        self.tenure_bucket_labels = ['<1 year', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5+ years']
        self.conctract_labels = ['Month-to-month', 'One year', 'Two year']
        
        #Categorical values that could be mapped to boolean values
        self.boolean_mapping = {'Yes': True, 'No': False, 'No internet service': False,  'No phone service': False, 0: False, 1:True, 'Male': True, 'Female': False, 'DSL':True, 'Fiber optic': True, 'No': False}
        
        #Categorize the features. The features can be sorted into key, numeric, categorical, and boolean features. Boolean can be further divided into service and demographic features.
        self.column_types = {
            'key': ['customerID'], #Unique ID, used to identify the costumer
            'numerical': ['Tenure', 'TotalCharges', 'MonthlyCharges'], #Features with numerical values (int/float)
            'categorical': ['PaymentMethod'], 
            'ordinal': {'Contract': self.conctract_labels},
            'boolean': ['SeniorCitizen', 'PaperlessBilling', 'Dependents', 'Churn', 'MultipleLines', 'OnlineSecurity', 'Male', 'OnlineBackup', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'DeviceProtection', 'InternetService', 'PhoneService', 'Partner'],
            'demographic': ['SeniorCitizen', 'PaperlessBilling', 'Dependents', 'Churn', 'Male', 'Partner'],
            'service': ['OnlineBackup', 'OnlineSecurity', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'DeviceProtection', 'InternetService', 'PhoneService', 'MultipleLines']
        }


    def preprocess_data(self, cutoff_percentage = 5):
        '''Cleans and orders the data for further use'''

        #Renames some of the columns for easier use
        self.df = self.df.rename(columns={'gender': 'Male', 'tenure':'Tenure'})

        self.df['TotalCharges'] = self.df['TotalCharges'].replace(' ', '0').astype(float) #Fixes an issue where fresh costumer has an empty string rather than 0 as total cost
        
        cutoff = cutoff_percentage * self.df.shape[0] // 100
        if self.df.isnull().sum().sum() > cutoff:
            print('Missing values exceed cutoff!')
        
        self.boolean_encoding(self.boolean_mapping)

    def feature_engineering(self):
        '''Creates new features based on existing features'''

        self.df['NumServices'] = self.df[self.column_types["service"]].sum(axis=1)
        self.df['AverageCharges'] = self.df['TotalCharges']/(self.df['Tenure']+1) #Average charge per customer
        self.df['TenureBucket'] = pd.cut(
            self.df['Tenure'], bins=[-1, 12, 24, 36, 48, 60, float('inf')],
            labels=self.tenure_bucket_labels).astype('string')
        self.df['TenureContract'] = self.df['TenureBucket'] + self.df['Contract']
        self.df['HighPaying'] = ((self.df['MonthlyCharges'] > self.df['MonthlyCharges'].median()).astype(int)).astype(bool)
        self.df['CostPerService'] = self.df['AverageCharges'] / (self.df['NumServices'] + 1)
        self.df['ServiceRatio'] = self.df['NumServices']/len(self.column_types["service"])
        self.df['PricePerTenureMonth'] =(self.df['TotalCharges']-self.df['MonthlyCharges'])/(self.df['Tenure']+1)
        self.df['TenureMonthlyCharge'] = self.df['Tenure'] * self.df['MonthlyCharges']
        self.df['TenureNumServices'] = self.df['Tenure'] * self.df['NumServices']
        self.df['Contract'] = self.df['Contract'].astype('category')
        self.df['PaymentMethod'] = self.df['PaymentMethod'].astype('category')
        self.df['TenureBucket'] = self.df['TenureBucket'].astype('category')
        self.df['TenureContract'] = self.df['TenureContract'].astype('category')

        #Updates the feature category sets with new features
        self.column_types["numerical"].append('NumServices')
        self.column_types["numerical"].append('AverageCharges')
        self.column_types["ordinal"]['TenureBucket'] = self.tenure_bucket_labels
        self.column_types["ordinal"]["TenureContract"] = [f"{t}{c}" for t in self.tenure_bucket_labels for c in self.conctract_labels]
        self.column_types["boolean"].append('HighPaying')
        self.column_types["numerical"].append('CostPerService')
        self.column_types["numerical"].append('ServiceRatio')
        self.column_types["numerical"].append('PricePerTenureMonth')
        self.column_types["numerical"].append('TenureMonthlyCharge')
        self.column_types["numerical"].append('TenureNumServices')


    def append_new_feature(self, feature, function):
        '''Appends a new feature to the dataframe based on a function'''

        self.df[feature] = function(self.df)
        self.column_types["numerical"].append(feature)


    def data_processing_and_feature_engineering(self):
        '''Cleans, categorizes, processes, and engineers the data for machine learning, and saves to a new file for quicker retrival'''

        self.categorize_features()
        self.data_cleaning()
        self.feature_engineering()

        #Saves the new data to file for quicker retrival
        self.df.to_csv('Telco-Customer-Churn-new.csv', index=False)
        with open('column_types.json', 'w') as outfile: 
            json.dump(self.column_types,outfile, indent=4,sort_keys=True)

    #╔════════════════════════════════════════════════════════════════════╗
    #║                EXPLORATION, CLEANING, & ENGINEERING                ║
    #╚════════════════════════════════════════════════════════════════════╝
    def data_exploration(self):
        '''Performs exploratory data analysis'''

        self.txt_output.append('DATA EXPLORATION\n')

        self.get_data_shape()
        self.get_unique_values()
        self.get_dtypes()
        self.compute_binary_feature_ratios()
        self.calculate_numerical_statistics()
        self.preview_all_columns()


    def get_data_shape(self):
        '''Provides an overview of the data set'''

        self.txt_output.append('Number of features: ' + str(self.df.shape[1]))
        self.txt_output.append('Number of entries: ' + str(self.df.shape[0]) + "\n")

    def get_unique_values(self):
        '''Prints the unique values of each feature'''

        self.txt_output.append("Unique values for each feature:")
        for col in self.df:
            self.txt_output.append(col + f'({self.df[col].nunique()}): ' + str(self.df[col].unique()))

    def get_dtypes(self):
        '''Prints the datatype of each feature'''

        self.txt_output.append("\nDatatype for each feature:\n")
        self.txt_output.append(str(self.df.dtypes))

    def compute_binary_feature_ratios(self):
        '''Calculates the ratio of boolean values'''

        self.txt_output.append("\nRatios for boolean features:")
        for col in self.df[self.column_types["boolean"]]:
            self.txt_output.append(f'{col}: ' + str(round(self.df[col].mean(),2)))

    def calculate_numerical_statistics(self):
        '''Calculates the metrics for numerical features'''

        self.txt_output.append("\nMetrics of numerical features:\n")

        for col in self.column_types['numerical']:
            self.txt_output.append(str(self.df[col].describe()) + "\n")

    def preview_all_columns(self):
        '''Shows the first five feature and dtype of every feature in the data set'''

        self.txt_output.append("\nThe five first entries of each feature:\n")

        for feature in self.df.columns:
            self.txt_output.append(str(self.df[feature].head()) + "\n")
        




    def feature_exploration(self, plot = False):
        '''Explores the data set in depth, analysing for patterns and insights.'''

        tenure_by_churn = self.df.groupby("Churn")["Tenure"].mean().reset_index()
        churn_by_contract = self.df.groupby("Contract")["Churn"].mean().reset_index()
        churn_by_payment_method = self.df.groupby("PaymentMethod")["Churn"].mean().reset_index()
        high_paying_churn = self.df.groupby("HighPaying")["Churn"].mean().reset_index()
        churn_by_payment_contract = self.df.groupby(["PaymentMethod", "Contract"])["Churn"].mean().unstack()
        tenure_bucket_churn = self.df.groupby("TenureBucket")["Churn"].mean().reset_index()

        corr_matrix = self.df[self.column_types['numerical']].corr()

        self.txt_output.append('\nFEATURE EXPLORATION')
        self.txt_output.append("\nAverage tenure of churned and non-churned costumers:")
        self.txt_output.append(str(tenure_by_churn))
        self.txt_output.append("\nChurn by contract:")
        self.txt_output.append(str(churn_by_contract))
        self.txt_output.append("\nChurn by payment method:")
        self.txt_output.append(str(churn_by_payment_method))
        self.txt_output.append("\nChurn by high paying customers:")
        self.txt_output.append(str(high_paying_churn))
        self.txt_output.append("\nChurn by tenure bucket:")
        self.txt_output.append(str(tenure_bucket_churn))
        self.txt_output.append("\nChurn by Payment Method and Contract:")
        self.txt_output.append(str(churn_by_payment_contract))
        self.txt_output.append("\nCorrelation Matrix:")
        self.txt_output.append(str(corr_matrix))
      

        if plot:
            featureplots = [
                PlotBuilder().set_title("Average tenure of churned and non-churned costumers").set_rotation(40).set_x_label(" ").set_tick_labels(["Churned", "Non-churned"]).create_barplot(tenure_by_churn,"Churn","Tenure"),
                PlotBuilder().set_title("Churn rate by contract").set_rotation(40).set_limits(y_lim=(0,1)).create_barplot(churn_by_contract,"Contract","Churn"),
                PlotBuilder().set_title("Churn rate by payment method").set_rotation(40).set_limits(y_lim=(0,1)).create_barplot(churn_by_payment_method,"PaymentMethod","Churn"),
                PlotBuilder().set_title("Churn rate by high paying customers").set_rotation(40).set_limits(y_lim=(0,1)).create_barplot(high_paying_churn,"HighPaying","Churn"),
                PlotBuilder().set_title("Churn rate by payment method and contract").create_heatmap(churn_by_payment_contract),
                PlotBuilder().set_title("Tenure Distrubution by Churn").create_histogram(self.df[["Tenure","Churn"]], "Tenure", "Churn"),
                PlotBuilder().set_title("Churn by Tenure Bucket").set_rotation(40).set_limits(y_lim=(0,1)).create_barplot(tenure_bucket_churn,"TenureBucket","Churn",order=self.column_types["ordinal"]['TenureBucket']),
                PlotBuilder().set_title("Correlation Matrix").set_title("Feature Correlation Matrix").set_rotation(40).create_heatmap(corr_matrix)
            ]

            for plot in featureplots:
                self.save_plot(plot)

    def plot_features(self, feature, target):
        '''Plots a feature against the target variable'''
        if feature in self.column_types['numerical']:
            PlotBuilder().set_title(f'{feature} vs {target}').create_scatterplot(self.df, feature, target)
        elif feature in self.column_types['categorical']:
            PlotBuilder().set_title(f'{feature} vs {target}').create_boxplot(self.df, feature, target)
        elif feature in self.column_types['ordinal'].keys():
            PlotBuilder().set_title(f'{feature} vs {target}').create_boxplot(self.df, feature, target)
        elif feature in self.column_types['boolean']:
            PlotBuilder().set_title(f'{feature} vs {target}').create_countplot(self.df, feature, target)


    def get_model_log(self,model):
        '''Returns the log of the model'''
        self.txt_output.append(f"\nMODEL: {model.model_type.upper()}")  

        if model.cv_results.any():
            self.txt_output.append("\nCross-validation Results:")
            self.txt_output.append(f"Individual fold scores: {model.cv_results}")
            self.txt_output.append(f"Mean CV score: {model.cv_results.mean():.3f}")
            self.txt_output.append(f"Std CV score: {model.cv_results.std():.3f}\n")

        if model.metrics:
            self.txt_output.append("\nModel Performance Metrics:")
            for name, value in model.metrics.items():
                self.txt_output.append(f'{name.capitalize()}: {round(value, 2)}')

        if model.conf_mat.any():
            self.txt_output.append("\nConfusion Matrix:")
            self.txt_output.append("                 Predicted")
            self.txt_output.append("                 Neg    Pos")
            self.txt_output.append(f"Actual  Neg     {model.conf_mat[0][0]:<6d} {model.conf_mat[0][1]:<6d}")
            self.txt_output.append(f"        Pos     {model.conf_mat[1][0]:<6d} {model.conf_mat[1][1]:<6d}\n")
    
        if model.feature_importance is not None:
            self.txt_output.append('\nTop 5 most important features:')
            self.txt_output.append(str(model.feature_importance.head()))
    
   
    #╔════════════════════════════════════════════════════════════════════╗
    #║                     MACHINE LEARNING MODELS                        ║
    #╚════════════════════════════════════════════════════════════════════╝
    def initialize_model_training(self,features,target,models, random_state=42):
        '''Initiates model training based on selected features and target'''

        X = self.df[features]
        y = self.df[target]

        self.txt_output.append('\nMODEL TRAINING\n')
        
        self.txt_output.append('FEATURES: ' + ', '.join(features))
        self.txt_output.append('TARGET: ' + target)

        modelbuild = ModelBuilder(X,y,self.column_types, random_state=random_state)
        for model in models:
            modelbuild.run_model(model)
            self.get_model_log(modelbuild)

    #╔════════════════════════════════════════════════════════════════════╗
    #║                   FILE PROCESSING AND LOGGING                      ║
    #╚════════════════════════════════════════════════════════════════════╝
    def read_file(self,path):
        self.log = open(path,'a')

    def save_plot(self,plotbuild):
        plotbuild.fig.savefig(f'plots\{plotbuild.title.replace(" ","_")}_{plotbuild.type}.png',bbox_inches="tight")

    #╔════════════════════════════════════════════════════════════════════╗
    #║                         HELPER FUNCTIONS                           ║
    #╚════════════════════════════════════════════════════════════════════╝
    def boolean_encoding(self, boolean_mapping):
        for col in self.column_types['boolean']:
            self.df[col] = self.df[col].map(boolean_mapping)

        
def main(path,save_log=False):
    '''Main function. Initiates everything'''

    #Initializes data from path:
    telco = Data(path,save_log)

    features = telco.df.drop(columns=['customerID','Churn','ServiceRatio','TenureNumServices','TenureContract','AverageCharges','TotalCharges','CostPerService','PricePerTenureMonth','TenureMonthlyCharge']).columns
    target = 'Churn'
    models = [RANDOM_FOREST, LOGISTIC_REGRESSION, XGBOOST]

    #telco.data_exploration()
    #telco.feature_exploration()
    #telco.initialize_model_training(features,target,models)

    telco.feature_exploration(plot=True)

    #Adds insights to a logfile
    if save_log:
        telco.read_file('Customer_Churn_log.txt')
        telco.log.write("\n".join(telco.txt_output) + '\n\n')
        telco.log.close()
        

if __name__ == '__main__':
    main('Telco-Customer-Churn-new.csv',save_log = False)