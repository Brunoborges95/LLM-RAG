from langchain.schema import Document
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEPubLoader,
)
import boto3
from typing import List, Any
import pathlib
import boto3

aws_textract = boto3.client("textract")


class Extract:
    def __init__(self, tolerance=0.01, vertical_diff=0.1):
        self.tolerance = tolerance
        self.vertical_diff = vertical_diff

    def _sort_key_with_tolerance(self):
        def key(box):
            top = box["bbox"]["Top"]
            left = box["bbox"]["Left"]
            return (round(top / self.tolerance), left)

        return key

    def textract(self, image_bytes):
        response = aws_textract.analyze_document(
            Document={"Bytes": image_bytes}, FeatureTypes=["TABLES", "FORMS"]
        )
        blocks = response["Blocks"]
        lines = []
        for block in blocks:
            if block["BlockType"] == "LINE":
                lines.append(
                    {"text": block["Text"], "bbox": block["Geometry"]["BoundingBox"]}
                )
        lines = sorted(lines, key=self._sort_key_with_tolerance())
        organized_text = []
        current_top = -1
        for line in lines:
            if (
                current_top == -1
                or abs(line["bbox"]["Top"] - current_top) > self.vertical_diff
            ):
                organized_text.append(line["text"])
                current_top = line["bbox"]["Top"]
            else:
                organized_text[-1] += " " + line["text"]
        extracted_text = "___".join(organized_text)
        return extracted_text


class EpubReader(UnstructuredEPubLoader):
    def __init__(self, file_path: str | list[str], **kwargs: Any):
        super().__init__(file_path, **kwargs, mode="elements", strategy="fast")


class DocumentLoaderException(Exception):
    pass


class DocumentLoader:
    """Loads in a document with a supported extension."""

    supported_extensions = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".epub": EpubReader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
    }


class Load:
    def __init__(self, temp_filepath: str, extractor: Extract = None):
        self.temp_filepath = temp_filepath
        self.extractor = extractor

    def load_document(self, OCR: bool) -> List[Document]:
        if OCR:
            if not self.extractor:
                raise ValueError("Extractor must be provided for OCR")
            with open(self.temp_filepath, "rb") as image_file:
                image_bytes = image_file.read()
                extracted_text = self.extractor(image_bytes)
            docs = [Document(page_content=extracted_text)]
        else:
            ext = pathlib.Path(self.temp_filepath).suffix
            loader_class = DocumentLoader.supported_extensions.get(ext)
            if not loader_class:
                raise DocumentLoaderException(
                    f"Invalid extension type {ext}, cannot load this type of file"
                )
            if loader_class == TextLoader:
                loader = loader_class(self.temp_filepath, encoding="UTF-8")
            else:
                loader = loader_class(self.temp_filepath)
            docs = loader.load()
        return docs
