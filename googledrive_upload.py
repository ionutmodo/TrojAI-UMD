import os, sys
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()

drive = GoogleDrive(gauth)

file_name = sys.argv[1]

file = drive.CreateFile({'title': f'test/{os.path.basename(file_name)}'})
file.SetContentFile(file_name)
file.Upload()
