import pandas as pd
from BackTranslation import BackTranslation
import seaborn as sns
import matplotlib.pyplot as plt
from time import sleep

#If you get "AttributeError: 'Translator' object has no attribute 'raise_Exception'", change your IP adress using a VPN.
#This implements back translation to all the dataset, in order to do it independently for X_train and X_test you need to edit the main function 

CSV_Path ="../data/DataSet (Augmentation.test).csv"
LANG = 'en'
OLANG = 'es'
API = 'google'

def translate(df, LANG, OLANG):
    translations = {}
    trans = BackTranslation()
    
    df_en = df.copy()
    trans.raise_Exception = True

    for element in df_en['Item (Texto)'].unique():
        sleep(1)
        try:
            translations[element] =  trans.translate(element, src=OLANG, tmp = LANG).result_text
        except TypeError:
            translations[element] = element
    df_en.replace(translations, inplace = True)
    
    return df_en
    
def main():
    df = pd.read_csv(CSV_Path, header = 0)
    df['GravedadMode'] = df['Gravedad'].str.split(',',expand=True).mode(axis=1, numeric_only=False, dropna=True)[0]
    df['SesgoMode'] = df['Sesgo'].str.split(',',expand=True).mode(axis=1, numeric_only=False, dropna=True)[0]
    df = df[['Item (Texto)', 'GravedadMode', 'SesgoMode']]
    
    df_concat = pd.concat([df, translate(df, LANG, OLANG)], ignore_index=True, sort=False)
    df_concat.to_csv(f'clasificacion-{API}-{LANG}.csv')
    sns.catplot(data = df_concat, x = 'GravedadMode', kind = 'count')
    plt.show()
    
if __name__ == '__main__':
    main()
