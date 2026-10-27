import html
from pathlib import Path

import streamlit as st

from document_app_service import (
    DocumentIngestionService,
    QASessionService,
    save_uploaded_pdf,
)
from knowledge_base import KnowledgeBase
from config import ENABLE_DEV_TOOLS
from diagnostics import setup_issues


st.set_page_config(page_title="Persian Document Understanding", layout="wide")


def inject_bidi_styles():
    st.markdown(
        """
        <style>
        textarea {
            direction: rtl;
            text-align: right;
            unicode-bidi: plaintext;
            font-family: Tahoma, Arial, sans-serif;
            line-height: 1.8;
        }
        .bidi-text {
            direction: rtl;
            text-align: right;
            unicode-bidi: plaintext;
            font-family: Tahoma, Arial, sans-serif;
            line-height: 1.9;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
        }
        .typing-preview {
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 6px;
            padding: 0.75rem;
            min-height: 3rem;
            background: rgba(250, 250, 250, 0.6);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_bidi_text(text, extra_class=""):
    safe_text = html.escape(text or "")
    st.markdown(
        f'<div dir="auto" class="bidi-text {extra_class}">{safe_text}</div>',
        unsafe_allow_html=True,
    )


def format_citation_location(item):
    location = f'Page {item["page_number"]}'
    paragraph_span = item.get("paragraph_span") or item.get("paragraph_index")
    if paragraph_span:
        location += f" | Paragraph {paragraph_span}"
    return location


def format_region_location(region):
    region_type = {
        "image": "Image",
        "table": "Table",
        "formula": "Formula",
    }.get(region.get("region_type"), "Visual region")
    return f'Page {region["page_number"]} | {region_type} #{region["region_index"]}'


@st.cache_resource
def get_knowledge_base():
    return KnowledgeBase()


@st.cache_resource
def get_ingestion_service():
    return DocumentIngestionService(get_knowledge_base())


@st.cache_resource
def get_qa_service():
    return QASessionService(get_knowledge_base())


def render_region_caption_gallery(document_id):
    regions = get_knowledge_base().load_regions(document_id)
    if not regions:
        st.info("No image captions found.")
        return

    columns = st.columns(3)
    for index, region in enumerate(regions):
        with columns[index % len(columns)]:
            st.markdown(f"**{format_region_location(region)}**")
            crop_path_value = region.get("crop_path")
            crop_path = Path(crop_path_value) if crop_path_value else None
            if crop_path and crop_path.is_file():
                st.image(str(crop_path), use_container_width=True)
            else:
                st.warning("Crop image file is missing.")
            render_bidi_text(region.get("caption") or "No caption generated.")


def render_document_table(documents):
    if not documents:
        st.info("No documents ingested yet.")
        return

    header = st.columns([3, 1, 1, 1, 1, 2, 1])
    header[0].markdown("**File**")
    header[1].markdown("**Status**")
    header[2].markdown("**Pages**")
    header[3].markdown("**Chunks**")
    header[4].markdown("**Vectors**")
    header[5].markdown("**Updated**")
    header[6].markdown("**Action**")

    for doc in documents:
        row = st.columns([3, 1, 1, 1, 1, 2, 1])
        row[0].write(doc["file_name"])
        row[1].write(doc["status"])
        row[2].write(doc["page_count"])
        row[3].write(doc["chunk_count"])
        row[4].write(doc.get("vector_status", "not_indexed"))
        row[5].write(doc["updated_at"])

        if row[6].button(
            "Delete",
            key=f"delete-{doc['document_id']}",
            type="secondary",
        ):
            result = get_ingestion_service().delete_document(doc["document_id"])
            message = result["message"]
            if result.get("vector_error"):
                message += " Qdrant vector cleanup failed: " + result["vector_error"]
            st.session_state["_operation_notice"] = {
                "warning": not result["deleted"] or bool(result.get("vector_error")),
                "message": message,
            }
            st.rerun()

        if doc["error"]:
            st.error(doc["error"])
        if doc.get("vector_error"):
            st.warning(doc["vector_error"])

        with st.expander(
            f"Images and captions | {doc['file_name']}",
            expanded=False,
        ):
            render_region_caption_gallery(doc["document_id"])


def render_ingestion_tab():
    st.subheader("Ingestion")
    notice = st.session_state.pop("_operation_notice", None)
    if notice:
        (st.warning if notice["warning"] else st.success)(notice["message"])
    include_captions = st.checkbox("Generate visual captions", value=True)
    index_vectors = st.checkbox("Index for semantic search", value=True)
    issues = setup_issues("ingestion", include_visual_captions=include_captions)
    for issue in issues:
        st.warning(issue)

    uploaded_files = st.file_uploader(
        "PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Ingest selected documents", type="primary", disabled=not uploaded_files or bool(issues)):
        service = DocumentIngestionService(
            get_knowledge_base(), include_visual_captions=include_captions,
            index_vectors=index_vectors,
        )

        for uploaded_file in uploaded_files:
            with st.status(f"Processing {uploaded_file.name}", expanded=True) as status:
                pdf_path = save_uploaded_pdf(uploaded_file)
                progress = st.progress(0)

                def update_progress(page_number, page_count):
                    progress.progress(page_number / max(page_count, 1))
                    st.write(f"Page {page_number} of {page_count}")

                try:
                    result = service.ingest_pdf(
                        pdf_path,
                        original_name=uploaded_file.name,
                        progress_callback=update_progress,
                    )
                    for warning in result.get("warnings", []):
                        st.warning(warning)
                    if result["status"] == "skipped":
                        status.update(label=f"Skipped {uploaded_file.name}", state="complete")
                        st.info(result["message"])
                    else:
                        status.update(label=f"Ingested {uploaded_file.name}", state="complete")
                        st.success(result["message"])
                except Exception as exc:
                    status.update(label=f"Failed {uploaded_file.name}", state="error")
                    st.error(str(exc))

    st.divider()
    st.subheader("Knowledge Base")
    render_document_table(get_knowledge_base().list_documents())

    if ENABLE_DEV_TOOLS:
        st.divider()
        st.subheader("Dev purge")
        confirmed = st.checkbox("Delete all documents, files, and vectors")
        if st.button("Purge SQLite, files, and Qdrant", type="secondary", disabled=not confirmed):
            result = get_ingestion_service().purge_dev_data()
            st.session_state["_operation_notice"] = {
                "warning": bool(result.get("vector_error")),
                "message": f"Qdrant purge failed: {result['vector_error']}" if result.get("vector_error") else "Purged SQLite rows, app files, and Qdrant vectors.",
            }
            st.rerun()


def render_qa_tab():
    st.subheader("QA Session")
    for issue in setup_issues("qa"):
        st.warning(issue)

    documents = get_knowledge_base().list_documents(ready_only=True)
    if not documents:
        st.info("Ingest at least one document before starting QA.")
        return

    doc_options = {
        f'{doc["file_name"]} | {doc["page_count"]} pages | {doc["updated_at"]}':
        doc["document_id"]
        for doc in documents
    }
    selected_names = st.multiselect(
        "Documents for this QA session",
        options=list(doc_options.keys()),
    )
    selected_ids = [doc_options[name] for name in selected_names]
    selected_docs = [
        doc for doc in documents if doc["document_id"] in set(selected_ids)
    ]

    question = st.text_area(
        "Question",
        value="آتروفی لوب گیجگاهی کی پیش می اید؟",
        height=90,
        help="Persian and embedded English technical terms are supported.",
    )

    retrieval_mode = st.selectbox(
        "Retrieval mode",
        options=["Text search (BM25)", "SimSearch (Qdrant)"],
        index=0,
        help="BM25 is lexical search. SimSearch uses Qdrant and local E5 embeddings.",
    )
    retrieval_mode_key = (
        "simsearch" if retrieval_mode == "SimSearch (Qdrant)" else "text"
    )
    simsearch_disabled = False
    if retrieval_mode_key == "simsearch":
        for issue in setup_issues("semantic"):
            st.warning(issue)
            simsearch_disabled = True
        if not get_qa_service().simsearch_available():
            st.warning("Qdrant is not reachable at the configured URL.")
            simsearch_disabled = True

        not_indexed = [
            doc["file_name"]
            for doc in selected_docs
            if doc.get("vector_status") != "indexed"
        ]
        if not_indexed:
            st.warning(
                "Selected documents are not indexed for SimSearch: "
                + ", ".join(not_indexed)
            )
            simsearch_disabled = True

    top_k = st.slider("Evidence chunks", min_value=1, max_value=10, value=5)

    if st.button(
        "Ask",
        type="primary",
        disabled=simsearch_disabled or not selected_names or not question.strip(),
    ):
        with st.spinner("Generating answer from retrieved evidence..."):
            try:
                result = get_qa_service().answer(
                    selected_ids, question.strip(), top_k=top_k,
                    retrieval_mode=retrieval_mode_key,
                )
            except Exception as exc:
                st.error(f"Question answering failed: {exc}")
                return

        st.subheader("Answer")
        st.caption({"generated": "Generated answer", "extractive": "Source excerpts", "insufficient_evidence": "Insufficient evidence"}[result["answer_mode"]])
        for warning in result["warnings"]:
            st.warning(warning)
        render_bidi_text(result["answer"])

        st.subheader("Citations")
        if not result["evidence"]:
            st.info("No evidence found.")
            return

        for rank, item in enumerate(result["evidence"], start=1):
            title = (
                f'{rank}. {item["file_name"]} | '
                f'{format_citation_location(item)} | '
                f'Score {item["score"]:.3f} | Matches {item["match_count"]}'
            )
            with st.expander(title, expanded=rank == 1):
                st.markdown("**Text**")
                render_bidi_text(item.get("display_text") or item["combined_text"])


def main():
    inject_bidi_styles()
    st.title("Persian Document Understanding")

    page = st.segmented_control(
        "Workspace",
        options=["Ingestion", "QA"],
        default="Ingestion",
        selection_mode="single",
    )

    if page == "Ingestion":
        render_ingestion_tab()
    else:
        render_qa_tab()


if __name__ == "__main__":
    main()
