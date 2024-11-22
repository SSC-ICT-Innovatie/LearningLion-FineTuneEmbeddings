import pickle
import sqlite3
import os
# from embedding import Embedding
from DataFetcher.libraries.data_classes.range_enum import Range
from ingester.libraries.preprocessor import Preprocessor
# from ubiops_helper import UbiopsHelper

class Database:
  bm25Retriever = None
  vector_store = None
  vectordb_folder = "vectordb"
  vectordb_name = "NewPipeChroma"
  embeddings = None
  range = Range.Tiny
  
  def __init__(self, range=Range.Tiny):
        self.range = range
        print("Database class initialized")

  def getNameBasedOnRange(self, range=Range.Tiny):
        if self.range is not None:
            return (f"NewPipeChroma_{range.name}", f"vectordb_{range.name}")
  
  def apply_database_schema(self):
        # Apply schema to database
        con = self.get_database_connection()
        cursor = con.cursor()
        # Create the documents table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                UUID TEXT PRIMARY KEY,
                filename TEXT,
                subject TEXT,
                producer TEXT,
                content TEXT,
                summirized TEXT,
                document_type TEXT,
                document BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                api_upload_date TIMESTAMP
            );
            """
        )

        # Create the questions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS questions (
                UUID TEXT,
                QUESTIONNUMBER TEXT,
                question TEXT,
                answer TEXT,
                PRIMARY KEY(UUID, QUESTIONNUMBER),
                FOREIGN KEY(UUID) REFERENCES documents(UUID)
            );
            """
        )

        # Create the footnotes table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS footnotes (
                UUID TEXT,
                footnote_number TEXT,
                footnote TEXT,
                PRIMARY KEY(UUID, footnote_number),
                FOREIGN KEY(UUID) REFERENCES documents(UUID)
            );
            """
        )
        con.commit()
        print("Database schema applied")
        con.close()
  def insertDocument(self, uuid, filename, doc_subject, doc_producer, full_text, blobData, summirized, questions, answers, footnotes, apiUploadDate):
        con = self.get_database_connection()
        # Check if document already exists
        results = con.execute("SELECT * FROM documents WHERE UUID=?", (uuid,)).fetchall()
        if len(results) > 0:
            print(f"Document with UUID {uuid} already exists in database")
        else:
            pre = Preprocessor()
            # Insert document
            con.execute("INSERT INTO documents (UUID, filename, subject, producer, content, summirized, document_type, document, api_upload_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                                (uuid, filename, doc_subject, doc_producer, full_text, summirized, "pdf", blobData, apiUploadDate))
            for footnote in footnotes:
                footnoteNumber = pre.get_footnote_number(footnote)
                con.execute("INSERT INTO footnotes (UUID, footnote_number, footnote) VALUES (?, ?, ?)",
                                 (uuid, footnoteNumber, footnote))
            
            print("Footnotes written to database")
            print(f"questions: {questions}") 
            for i in range(len(questions)):
                question = questions[i]
                answer = answers[i]
                questionNumber = pre.get_question_number(question)[0]
                con.execute("INSERT INTO questions (UUID, QUESTIONNUMBER, question, answer) VALUES (?, ?, ?, ?)",
                                 (uuid, questionNumber, question, answer))
            con.commit()
            print("Written to database")
            con.close()

  def get_database_connection(self) -> sqlite3.Connection:
        con = None
        print("No database connection set")
        if self.vectordb_name is not None:
            names = self.getNameBasedOnRange(self.range)
            print(f"Connecting to database {names[0]}")
            con = sqlite3.connect(f"{names[0]}.db", detect_types=sqlite3.PARSE_DECLTYPES)
            print(f"Database connection set to {self.vectordb_name}")
        else:
            print("Database connection already set")
        return con
    