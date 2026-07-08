import mcp_server.tools as t
t._VECTOR_STORE = None
vs = t._get_vector_store()
print(f'Ingested: {vs._collection.count()} chunks')