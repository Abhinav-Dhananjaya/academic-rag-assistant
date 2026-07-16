from qdrant_client import QdrantClient
client = QdrantClient(path="./local_qdrant")
print(client.get_collections())
