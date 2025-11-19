-- Create table for storing code embeddings with metadata
CREATE TABLE IF NOT EXISTS code_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_extension TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL, -- For duplicate detection
    language TEXT NOT NULL, -- programming language
    chunk_index INTEGER NOT NULL, -- order of chunks within file
    total_chunks INTEGER NOT NULL, -- total chunks in file
    embedding vector(1536), -- OpenAI embeddings dimension
    token_count INTEGER, -- token count for cost tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    UNIQUE(file_path, chunk_index),
    UNIQUE(content_hash),
    CHECK (chunk_index >= 0),
    CHECK (total_chunks > 0)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_code_embeddings_file_path ON code_embeddings(file_path);
CREATE INDEX IF NOT EXISTS idx_code_embeddings_language ON code_embeddings(language);
CREATE INDEX IF NOT EXISTS idx_code_embeddings_file_extension ON code_embeddings(file_extension);
CREATE INDEX IF NOT EXISTS idx_code_embeddings_content_hash ON code_embeddings(content_hash);
CREATE INDEX IF NOT EXISTS idx_code_embeddings_created_at ON code_embeddings(created_at);

-- Vector similarity search index (IVFFlat for good performance)
CREATE INDEX IF NOT EXISTS idx_code_embeddings_embedding
ON code_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_code_embeddings_updated_at
    BEFORE UPDATE ON code_embeddings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
