import pandas as pd
import translators as ts
import seaborn as sns


CSV_Path ="../data/clasificacion.csv"
LANG = 'en'
OLANG = 'es'

def translate(df, LANG, OLANG):
    translations = {}
    back_translations = {}
    
    df_en = df.copy()

    for element in df_en['Item (Texto)'].unique():
        translations[element] =  ts.google(element, OLANG, LANG)
    df_en.replace(translations, inplace = True)

    for element in df_en['Item (Texto)'].unique():
        back_translations[element] =  ts.google(element, LANG, OLANG)
    df_en.replace(back_translations, inplace = True)
    
    return df_en
    
def main():
    df = pd.read_csv(CSV_Path, header = 0)
    df['GravedadMode'] = df['Gravedad'].str.split(',',expand=True).mode(axis=1, numeric_only=False, dropna=True)[0]
    df['SesgoMode'] = df['Sesgo'].str.split(',',expand=True).mode(axis=1, numeric_only=False, dropna=True)[0]
    df = df[['Item (Texto)', 'GravedadMode', 'SesgoMode']]
    df_en = translate(df, LANG, OLANG)
    df_concat = pd.concat([df, df_en], ignore_index=True, sort=False)
    df_concat.to_csv(f'clasificacion-{google}-{LANG}.csv')
    sns.catplot(data = df, x = 'GravedadMode', kind = 'count')
    plt.show()
    
if __name__ == '__main__':
    main()
