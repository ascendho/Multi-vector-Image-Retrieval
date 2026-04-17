# Run Qdrant with storage directory mapped to
# a corresponding host directory
docker run \
  -p "6333:6333" -p "6334:6334" \
  -e "QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=1" \
  -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
  "qdrant/qdrant:v1.15.4"
