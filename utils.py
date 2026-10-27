from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from pdf2image import convert_from_path



@contextmanager
def rendered_pdf_pages(pdf_path):
    """Render to owned temporary storage and clean it even when ingestion fails."""
    with TemporaryDirectory(prefix="persian-qa-render-") as directory:
        paths = convert_from_path(
            str(pdf_path), output_folder=directory, fmt="jpeg", paths_only=True,
            thread_count=1,
        )
        yield [Path(path) for path in paths]
