from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    get_response_synthesizer,
    VectorStoreIndex
)
import os
import s3fs 

# Initialize S3 filesystem
s3_bucket = 's3://secure-s-ai-dl-prod'

fs = s3fs.S3FileSystem(secret="", key="")
fs.ls(s3_bucket)
from llama_index.indices.document_summary import DocumentSummaryIndex

os.environ.setdefault("OPENAI_API_KEY", "")

# Replace the local file path with the S3 path
documents = SimpleDirectoryReader(input_files=["indexdata.txt"]).load_data()

response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize"
)
service_context = ServiceContext.from_defaults()
doc_summary_index = DocumentSummaryIndex.from_documents(
    documents,
    service_context=service_context,
    response_synthesizer=response_synthesizer,
    show_progress=True,
)
from llama_index.indices.loading import load_index_from_storage
from llama_index import StorageContext
import os
from llama_index.response_synthesizers import TreeSummarize
from pprint import pprint
os.environ.setdefault("OPENAI_API_KEY", "")
storage_context = StorageContext.from_defaults(persist_dir="Large-Summary-Index")
doc_summary_index = load_index_from_storage(storage_context)


query_engine = doc_summary_index.as_query_engine(
    response_mode="tree_summarize", use_async=True
)

pprint(query_engine.query("Summarize the document"))