import pandas as pd
from BackTranslation import BackTranslation
from time import sleep
from sklearn.model_selection import train_test_split

#If you get "AttributeError: 'Translator' object has no attribute 'raise_Exception'", change your IP adress using a VPN.
#This is meant to be used after nlpaug
#This implements back translation only to the training split, and the result is two dataframes, one for testing and one for training


CSV_Path ="nlpaug_data.csv"
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
    
def maind():
    df = pd.read_csv(CSV_Path, header = 0)
    x = df['Item (Texto)'].values
    y = df['GravedadMode'].values
    
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = 0)
    df2 = pd.DataFrame(columns = ['Item (Texto)' , 'GravedadMode'])
    df3 = pd.DataFrame(columns = ['Item (Texto)' , 'GravedadMode'])
    for i in range(len(X_train)):
        df2.loc[i] = [X_train[i],y_train[i]]
    for i in range(len(X_test)):
        df3.loc[i] = [X_test[i],y_test[i]]
    df_concat = pd.concat([df2, translate(df2, LANG, OLANG)], ignore_index=True, sort=False)
    df_concat.to_csv(f'Train-{API}-{LANG}.csv')
    df3.to_csv(f'Test-{API}-{LANG}.csv')
    
if __name__ == '__main__':
    maind()
