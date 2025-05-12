from flask import Flask, request, jsonify, render_template
import os
import requests
import numpy as np
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.text_splitter import SentenceSplitter
from dotenv import load_dotenv
import time

load_dotenv()
#os.getenv("OPENAI_API_KEY")

app = Flask(__name__, template_folder='templates', static_folder='static')

DATA_FOLDER = "data"

text_splitter = SentenceSplitter(
        chunk_size=256,
        chunk_overlap=20,
        paragraph_separator="\n\n"
    )

Settings.text_splitter = text_splitter

pipeline = IngestionPipeline(
        transformations=[text_splitter],
        # Optional vector store (if you want to use a specific one)
        # vector_store=ChromaVectorStore(),
    )

documents = SimpleDirectoryReader(DATA_FOLDER).load_data()

start_time = time.time()
nodes = pipeline.run(documents=documents)
print(f"Document processed into {len(nodes)} nodes in {time.time() - start_time:.2f} seconds")

index = VectorStoreIndex.from_documents(
    documents, transformations=[text_splitter]
)

batch_size = 100
total_batches = (len(nodes) - 1) // batch_size + 1

for i in range(0, len(nodes), batch_size):
  batch_end = min(i + batch_size, len(nodes))
  print(f"Indexing batch {i//batch_size + 1}/{total_batches} (nodes {i}-{batch_end})")
  index.insert_nodes(nodes[i:batch_end])
  print(f"Index creation completed in {time.time() - start_time:.2f} seconds total")

index.storage_context.persist(persist_dir="persisted_data")

query_engine = index.as_query_engine()

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        question = data.get('question', '')
        response = query_engine.query(question)
        return jsonify({'response': str(response)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


