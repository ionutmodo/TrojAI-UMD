import os, sys
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()

drive = GoogleDrive(gauth)

file_name = os.path.basename(sys.argv[1])
print(f'File name on Google Drive: {file_name}')

file = drive.CreateFile({'title': file_name})
file.SetContentFile(file_name)
file.Upload()
