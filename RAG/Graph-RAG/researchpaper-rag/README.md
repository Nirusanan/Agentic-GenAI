# Research Paper Assistant - Graph RAG

step 1: Install GraphRAG
```
pip install graphrag
```

step 2: Initialize workspace
```
graphrag init --root ./researchpaper-rag
```

step 3: add api key in .env and config settings.yaml

step 4: Indexing
```
graphrag index --root ./researchpaper-rag
```

### After indexing, you got this result
![indexing](https://github.com/user-attachments/assets/c6683f1a-6db1-40d9-ac02-a2c5665c0f37)

Ask the question using local search:
```
graphrag query --root ./researchpaper-rag --method local --query "Where we use time-series tasks?"
```
![answer](https://github.com/user-attachments/assets/04f7f2e2-0424-4587-8a63-65382d4dc521)


### Visualize the knowledge graph
![graphrag-knowledge](https://github.com/user-attachments/assets/e5bb8ccd-e0aa-4ea4-94bb-049b28c52b75)

