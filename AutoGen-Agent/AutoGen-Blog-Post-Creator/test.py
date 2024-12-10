import chromadb
from PyPDF2 import PdfReader

client = chromadb.PersistentClient(path='/tmp/chromadb')
collection = client.get_collection("pdfreader")

# Retrieve the first few chunks
retrieved_docs = collection.get(limit=30)
for doc in retrieved_docs["documents"]:
    print(doc)
