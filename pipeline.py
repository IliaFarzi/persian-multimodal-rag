class MultimodalPipeline:
    def __init__(self, ocr, captioner, preprocessor, chunker, retriever):
        self.ocr = ocr
        self.captioner = captioner
        self.preprocessor = preprocessor
        self.chunker = chunker
        self.retriever = retriever

    def process(self, image_paths):
        all_chunks = []

        for image_path in image_paths:
            # OCR
            text = self.ocr.extract_text(image_path)
            text = self.preprocessor.normalize(text)

            # Caption
            caption = self.captioner.generate_caption(image_path)

            # Combine
            combined = f"{text}\nتوضیح تصویر: {caption}"

            # Chunk
            chunks = self.chunker.chunk(combined)
            all_chunks.extend(chunks)

        # Build BM25 index
        self.retriever.build_index(all_chunks)

        return all_chunks

    def query(self, query, top_k=3):
        return self.retriever.search(query, top_k)