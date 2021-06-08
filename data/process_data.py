import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load the data
    
    This function loads the data file and return a DataFrame
    
    Arguments:
        messages_filepath -> path to the message csv file
        categories_filepath -> path to the categories csv file
    Output:
        df -> Output DataFrame File
    """    
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(categories, messages, on=["id"], how="outer")
    columns_names = categories['categories'][0].split(';')
    columns_names = [x[:-2] for x in columns_names]
    
    categories2 = categories['categories'].str.split(';',expand=True)
    categories2 = categories2.applymap(lambda x: x[-1])
    categories2.columns = columns_names
    categories2 = categories2.join(categories['id'])
    
    df.drop(columns='categories',axis=1,inplace=True)
    df = pd.merge(categories2, df, on=["id"], how="outer")
    
    return df

def clean_data(df):
    """
    Clean the data
    
    This function cleans the data
    
    Arguments:
        df -> Input DataFrame File
    Output:
        df -> Output DataFrame File
    """      
    df.drop_duplicates(inplace=True)
    #Remove child alone as it has all zeros only
    df = df.drop(['child_alone'],axis=1)
    
    # Given value 2 in the related field are neglible so it could be error. Replacing 2 with 1 to consider it a valid response.
    # Alternatively, we could have assumed it to be 0 also. In the absence of information I have gone with majority class.
    df.iloc[:,:-4] = df.iloc[:,:-4].applymap(int)
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    return df


def save_data(df, database_filename):
    """
    Save the data
    
    This function saves the data file e
    
    Arguments:
        df -> Input DataFrame File 
        database_filename -> database file name
    """       
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Disaster-Response', engine, index=False,if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()