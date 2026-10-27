# Third-party notices

The project source uses AGPL-3.0-only. This does not relicense third-party dependencies, model weights, or documents. Installations must retain the license information supplied with those components.

- **PyMuPDF / MuPDF**: Artifex Software; available under AGPL or a commercial agreement. This release uses the open-source stack. [License documentation](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright).
- **Gemma 3**: Google DeepMind, `google/gemma-3-4b-it`. No weights are redistributed by this repository. Download and use are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and [upstream model card](https://huggingface.co/google/gemma-3-4b-it). The project license does not replace those terms.
- **Multilingual E5 small**: `intfloat/multilingual-e5-small`, MIT model license. [Upstream model card and attribution](https://huggingface.co/intfloat/multilingual-e5-small). The model is used as an existing embedding model, not trained by this project.
- **Tesseract, Qdrant, PyTorch, Transformers, Streamlit, Hazm, Pillow, OpenCV, pdf2image, rank-bm25 and other installed packages**: retain their upstream licenses and attribution in the installed distributions. The dependency versions used by the application are listed in `requirements.txt`.
- **Evaluation papers**: third-party publications identified in `evaluation/corpus.json`. Redistribution permission has not been established. PDF files, page images, crops, OCR dumps, and the paper authors' original figures are excluded from the public source snapshot. Questions and short reference answers are project-authored annotations; citations identify the underlying papers. Raw evaluation outputs should be reviewed before sharing because they can contain extracted source passages.
- **Synthetic demonstration**: `examples/make_demo_pdf.py` creates original, invented demonstration material. Its example numbers must not be presented as measured QA performance.
