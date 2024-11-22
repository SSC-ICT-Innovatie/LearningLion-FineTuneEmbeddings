import zipfile
import pymupdf
import pymupdf4llm
import os
from ingester.libraries.database import Database
from ingester.libraries.preprocessor import Preprocessor

class Ingestion:
        
    def summirize(self, text):
        # TODO: Implement summarization
        return "Summarized text"
    
    def ingest(self, source_dir=None, database:Database|None =None):
        sourceDir = source_dir
        # if zip is downloaded, unzip it with shutil
        for filename in os.listdir(sourceDir):
                file_path = os.path.join(sourceDir, filename)
                
                if filename.endswith(".zip"):
                    print(f"Unzipping {filename}")
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(sourceDir)
                    print(f"Unzipped and removed {filename}")

        totalFiles_in_dir = len([name for name in os.listdir(sourceDir) 
                                if os.path.isfile(os.path.join(sourceDir, name)) and name.endswith('.pdf')])
        print(f"Total PDF files in directory found: {totalFiles_in_dir}")
        items = 0
        if os.path.exists(sourceDir):
            for filename in os.listdir(sourceDir):
                try:
                    if filename.endswith(".pdf"):
                        items += 1
                        file_path = os.path.join(sourceDir, filename)
                        with open(file_path, "rb") as pdf_file:
                            uuid = filename.split(".")[0]
                            reader = pymupdf.Document(pdf_file)
                            # Move the pointer back to the start of the file
                            pdf_file.seek(0)
                            # Read the raw bytes of the PDF document
                            blobData = pdf_file.read()
                            pdf_file.seek(0)
                            metadata_text = reader.metadata
                            full_text = ""
                            doc_subject = metadata_text.get('subject') or "unknown"
                            doc_producer = metadata_text.get('producer') or "unknown"
                            print(f"metadata: {metadata_text}")
                            apiUploadDate = metadata_text.get("creationDate")
                            full_text = pymupdf4llm.to_markdown(reader, margins=(0,0,0,0))
                            pre = Preprocessor()
                            # clean_full_text = pre.clean_text_MD(full_text)
                            
                            # heading = pre.get_heading(full_text)
                            qa_list = pre.get_question_and_answer(full_text)
                            
                            # ##
                            database.insertDocument(
                                uuid=uuid,
                                filename=filename,
                                doc_subject=doc_subject,
                                doc_producer=doc_producer,
                                full_text=full_text,
                                blobData=blobData,
                                summirized=self.summirize(full_text),
                                questions=qa_list[0],
                                answers=qa_list[1],
                                footnotes=pre.get_footnotes(full_text),
                                apiUploadDate=apiUploadDate
                            )
                            print(f"Processed {items} files out of {totalFiles_in_dir}")
                except Exception as e:
                    print(f"Error while processing file: {filename}")
                    print(f"Error: {e}")

        print("done")
        print(f"Total files: {items}")
        

def batch(iterable, batch_size):
    """Yield successive batches of specified size from the iterable."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]