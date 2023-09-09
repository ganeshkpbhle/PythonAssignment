#Importing Required Libraries
from sqlalchemy import create_engine,Column, Float, Integer, String, MetaData,inspect,text
from sqlalchemy.orm import sessionmaker,declarative_base
import pandas as pd
import urllib.parse
import numpy as np
import math,random
import matplotlib.pyplot as plt
import seaborn as sns
#setting up the Password and connection string for db connection
un = "root"
enc_pwd=urllib.parse.quote_plus("inspiron6626")
def NewDBSession(dbname):
    #Initializing connection for the db and session
    conn_str=f"mysql+pymysql://{un}:{enc_pwd}@localhost/{dbname}"
    dbengine = create_engine(conn_str) #setting up db engine using connection string
    dbconn = dbengine.connect() #connecting to the db
    #Initializing current session with db to load csv data 
    Session = sessionmaker(bind=dbengine)
    curr_sessn = Session()
    print("Session Active : ",curr_sessn.is_active)
    return (curr_sessn,dbconn)
# Creating parent class for ORM to database
BaseORM = declarative_base() #declaring base class for custom mapping from data frame
BaseORM.metadata.bind = dbengine
class DataSetBase(BaseORM):
    __abstract__ = True
    id = Column(Integer, primary_key=True, autoincrement=True)
    x = Column(Float, nullable=False)
BaseORM.metadata.create_all(dbengine)
#declaring child classes for custom mapping from data frame
def create_custom_model(table_name, columns):
    class DerivedDataSet(DataSetBase):
        __tablename__ = table_name
    # Dynamically add columns based on DataFrame columns
    for column_name in columns:
        if column_name not in ['id', 'x']:
            setattr(DerivedDataSet, column_name, Column(Float))
    # Create the table in the database
    BaseORM.metadata.create_all(dbengine, tables=[DerivedDataSet.__table__])
    return DerivedDataSet
#Custom exception class to handle errors for file loading
class FileLoadErr(Exception):
    pass
def findbestfit(trainingds,idealfuncds):
    bestfit = {}
    maxlse = float('-inf')
    for col1 in trainingds:
        minlse = float('inf')
        bestcol = None
        currcol1 = trainingds[col1].to_numpy()
        for col2 in idealfuncds:
            currcol2 = idealfuncds[col2].to_numpy()
            mse = np.mean((currcol1-currcol2)**2)
            if mse < minlse:
                minlse = mse
                bestcol = col2
        bestfit[col1]=(minlse,bestcol)
        if(minlse>maxlse):
            maxlse=minlse
        idealfuncds.drop(bestcol,axis=1)
    return (bestfit,maxlse)
def IdentifyMinDeviation(bestfit,maxlse):
    passdict={"X":[],"Y":[],"IdealFunctionColumn":[],"MinDeviation":[]}
    upplimit= math.sqrt(2) * maxlse
    for index,row in testds.iterrows():
        testvalue = float(row['y'])
        mindev = float('inf')
        col = None
        for key in bestfit.keys():
            dev = (testvalue-bestfit[key][0])**2
            if dev < mindev:
                mindev = dev
                col = bestfit[key][1]
        passdict["X"].append(int(row['x']))
        passdict["Y"].append(testvalue)
        passdict["IdealFunctionColumn"].append(col)
        passdict["MinDeviation"].append(mindev if mindev<upplimit else "Miss")
    return pd.DataFrame(passdict)
'''generic method to populate dataframe from CSV File 
   param fp : actual path of the file along with name 
   param cols: expected number of columns from CSV file
   returns a dataFrame
'''
def load_csv_file(fp, cols):
    try:
        data = pd.read_csv(fp)
        if set(data.columns) != set(cols):
            raise FileLoadErr(f"Invalid CSV file format: {fp}")
        return data
    except Exception as e:
        raise FileLoadErr(f"Error loading CSV file: {fp} - {e}")
'''method to read the csv file and load in DB
   param traindata : training data file name (CSV)
   param idealfunc : ideal functions dataset file name (CSV)
'''
def create_and_load_database(traindata, idealfunc,testds):
    # Load training data and ideal functions dataset from CSV files
    trainingds = load_csv_file(traindata, ['x'] + ['y' + str(i) for i in range(1,5)])
    idealfuncds = load_csv_file(idealfunc, ['x'] + ['y' + str(i) for i in range(1, 51)])
    testds = load_csv_file(testds, ['x','y'])
    curr_sessn,dbconn = NewDBSession("assignment")
    #Populate TrainingDataTable
    trainingTable = create_custom_model('trainingdt', trainingds.columns)
    for _, row in trainingds.iterrows():
        data = {'x': row['x']}
        for column_name in row.index:
            if column_name not in ['id', 'x']:
                data[column_name] = row[column_name]
        table_instance = trainingTable(**data)
        curr_sessn.add(table_instance)
    #Populate IdealFunctionsTable
    idealFuncTable = create_custom_model('idealfunctions', idealfuncds.columns)
    for _, row in idealfuncds.iterrows():
        data = {'x': row['x']}
        for column_name in row.index:
            if column_name not in ['id', 'x']:
                data[column_name] = row[column_name]
        table_instance = idealFuncTable(**data)
        curr_sessn.add(table_instance)
    #Populate TestDataTable
    TestDataTable = create_custom_model('testdata', testds.columns)
    for _, row in testds.iterrows():
        data = {'x': row['x']}
        for column_name in row.index:
            if column_name not in ['id', 'x']:
                data[column_name] = row[column_name]
        table_instance = TestDataTable(**data)
        curr_sessn.add(table_instance)
    curr_sessn.commit()
    curr_sessn.close()
    dbconn.close()
if __name__ == '__main__':
    trainingds = 'train.csv'
    idealfuncds = 'ideal.csv'
    testds = 'test.csv'
    create_and_load_database(trainingds, idealfuncds,testds)
    curr_sessn,dbconn=NewDBSession("assignment")
    trainingds = pd.read_sql("select * from trainingdt",con=dbconn)
    xcol = trainingds['x']
    trainingds = trainingds.drop(['id','x'],axis=1)
    idealfuncds = pd.read_sql("select * from idealfunctions",con=dbconn)
    idealfuncds = idealfuncds.drop(['id','x'],axis=1)
    testds = pd.read_sql("select * from testdata",con=dbconn)
    testds = testds.drop(['id'],axis=1)
    curr_sessn.close()
    dbconn.close()

    #linegraph for training dataset
    num_columns = len(trainingds.columns)
    colors = sns.color_palette("husl", n_colors=num_columns)
    # Shuffling the colors randomly
    random.shuffle(colors)

    plt.figure(figsize=(10, 6))
    # Creating subplots for each column
    for i, col in enumerate(trainingds.columns):
        plt.subplot(2, 2, i + 1) 
        plt.plot(xcol,trainingds[col],label=col,color=colors[i])
        plt.title(f'Distribution of {col}')
        plt.xlabel("X")
        plt.ylabel(col)
    plt.tight_layout()
    plt.show()

    #Calling the bestfit method and Identify Mininmum deviation Method and Writing Result to Datbase in a result table
    bestfit,maxlse = findbestfit(trainingds,idealfuncds)
    resultdf = IdentifyMinDeviation(bestfit,maxlse)
    print(resultdf)
    curr_sessn,dbconn=NewDBSession("assignment")
    resultdf.to_sql('result',con=dbconn)
    curr_sessn.close()
    dbconn.close()