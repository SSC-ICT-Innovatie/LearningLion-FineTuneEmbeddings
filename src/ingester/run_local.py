from DataFetcher.libraries.data_classes.range_enum import Range
from ingester.libraries import database, ingestion

def run_local_ingest_stores(range=Range.Tiny):
  print("Running Main class")
  # Initialize components
  data = database.Database(range=range)
  ingest = ingestion.Ingestion()
  print("Classes initialized")
  data.apply_database_schema()
  print("Perform ingestion")
  ingest.ingest(source_dir='./tmp', database=data)
  print("Ingested")