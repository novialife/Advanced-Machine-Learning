import os
import io
from datetime import datetime
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient
from azure.keyvault.secrets import SecretClient


# ONLY FOR LOCAL TESTING #######################################
os.environ['AZURE_TENANT_ID'] = '3db27ecc-1791-4dda-9b51-798adfa4a3ca'
os.environ['AZURE_CLIENT_ID'] = 'bce307e4-3e58-4588-9991-e32c97534d47'
os.environ['AZURE_CLIENT_SECRET'] = '_0z8Q~4I7ChvKAnv~vFLDkXXu1wQR~xN2ILjHbCl'
#############################################################



# Configuration
BLOB_account = 'novialifecontainer'
BLOB_container = 'novialifeblob'
BLOB_name = 'out.txt'

FS_fname = 'in.txt'

KV_account = 'novialifes'
KV_secret_name = 'blobcontainer'

# Print datetime and environment variables
print(f'{datetime.now()}')
print(f'This is an environment variable: {os.environ.get("public1")}')
print(f'This is a secret environment variable: {os.environ.get("private1")}')

# Authenticate with Azure
# (1) environment variables, (2) Managed Identity, (3) User logged in in Microsoft application, ...
AZ_credential = DefaultAzureCredential()

# Retrieve primary key for blob from the Azure Keyvault
KV_url = f'https://{KV_account}.vault.azure.net'
KV_secretClient = SecretClient(vault_url=KV_url, credential=AZ_credential)
BLOB_PrimaryKey = KV_secretClient.get_secret(KV_secret_name).value

# Set the BLOB client
BLOB_CONN_STR = f'DefaultEndpointsProtocol=https;AccountName={BLOB_account};AccountKey={BLOB_PrimaryKey};EndpointSuffix=core.windows.net'
BLOB_client = BlobClient.from_connection_string(conn_str=BLOB_CONN_STR, container_name=BLOB_container, blob_name=BLOB_name)

# Read text-file from mounted fileshare and write to BLOB
with open(f'mnt/{FS_fname}', 'rb') as f:
    dataBytesBuffer = io.BytesIO(f.read())
    dataBytesBuffer.seek(0)
    BLOB_client.upload_blob(dataBytesBuffer, overwrite=True)
    print(f'File successfully uploaded to blob')