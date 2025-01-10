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

Ask the question using local search:
```
graphrag query --root ./researchpaper-rag --method local --query "Where we use time-series tasks?"
```

### visualize the knowledge graph
