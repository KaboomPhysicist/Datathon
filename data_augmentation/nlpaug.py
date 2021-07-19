from numpy import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nlpaug.augmenter.word import synonym

#This script implements Synonym Augmenter, only to the rows with GravedadMode value equal to 0, the result is the augmented dataframe 
#This is meant to be used before the final augmentation

CSV_Path ="../data/DataSet (Augmentation.test).csv"

def augmen(df):
    augmentation = {}
    aug = synonym.SynonymAug(aug_src='wordnet', lang='spa')
    
    df_a = df.copy()
    df_a = df_a.drop(df_a[(df_a['GravedadMode']=='2')].index)
    df_a = df_a.drop(df_a[(df_a['GravedadMode']=='1')].index)
    df_a = df_a.drop(df_a[(df_a['GravedadMode']=='3')].index)

    for element in df_a['Item (Texto)'].unique():
        augmentation[element] =  aug.augment(element)

    df_a.replace(augmentation, inplace = True)
    
    return df_a

def main():
    df = pd.read_csv(CSV_Path, header = 0)
    df['GravedadMode'] = df['Gravedad'].str.split(',',expand=True).mode(axis=1, numeric_only=False, dropna=True)[0]
    df['SesgoMode'] = df['Sesgo'].str.split(',',expand=True).mode(axis=1, numeric_only=False, dropna=True)[0]
    df = df[['Item (Texto)', 'GravedadMode', 'SesgoMode']]
    
    df2 = pd.concat([df, augmen(df)], ignore_index=True, sort=False)
    df2.to_csv(f'nlpaug_data.csv')
    sns.catplot(data = df2, x = 'GravedadMode', kind = 'count')
    plt.show()
    
if __name__ == '__main__':
    main()
