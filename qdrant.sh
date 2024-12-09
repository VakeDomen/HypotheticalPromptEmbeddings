sudo docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    --ulimit nofile=10000:10000 qdrant/qdrant