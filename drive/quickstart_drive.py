from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import pandas as pd

def data_download():
    #Conexión al servidor de Google
    GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = '../drive/client_secrets.json'
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth() # Creates local webserver and auto handles authentication.

    drive = GoogleDrive(gauth)

    #Ubica el spreadsheet con su ID y lo descarga
    data_file = drive.CreateFile({'id' : '1ERFL1EeV4aL7xyGFc-zzFYSnEb6fpHx79P3IQhOLSXo'})
    data_file.GetContentFile('../data/IdentificacionSesgo.xlsx',
                            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    
    #Lee el spreadsheet con pandas y lo convierte a CSV
    file = pd.read_excel('../data/IdentificacionSesgo.xlsx', sheet_name='DataSet', engine='openpyxl')
    file.to_csv('../data/clasificacion.csv',index=None, header=True)
