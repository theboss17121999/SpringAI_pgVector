-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS hstore;

-- Create the vector store table
CREATE TABLE IF NOT EXISTS vector_store (
                                            id TEXT PRIMARY KEY,
                                            content TEXT,
                                            metadata JSONB,
                                            embedding VECTOR(768)
);

-- Create an HNSW index for cosine similarity search
CREATE INDEX IF NOT EXISTS vector_store_embedding_idx
    ON vector_store
        USING HNSW (embedding vector_cosine_ops);

