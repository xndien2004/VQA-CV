from .ocr_embedding import OCREmbeddingBuilder

def build_ocr_embedding(config):
    ocr_embedding = OCREmbeddingBuilder(config)
    return ocr_embedding