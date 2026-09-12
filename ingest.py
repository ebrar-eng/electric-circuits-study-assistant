import os
import glob
from pypdf import PdfReader
from foundry_local_sdk import Configuration, FoundryLocalManager
import db

EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BATCH_SIZE = 16


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def main():
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    conn = db.get_connection()
    db.init_db(conn)
    db.clear_db(conn)

    docs_dir = "documents"
    txt_files = glob.glob(os.path.join(docs_dir, "*.txt"))
    pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
    all_files = txt_files + pdf_files
    print(f"Toplam {len(all_files)} dosya bulundu. İndeksleniyor...\n")

    total_chunks = 0
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        print(f"\nİşleniyor: {file_name}")
        if file_path.endswith(".pdf"):
            full_text = extract_text_from_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read()

        chunks = chunk_text(full_text)
        if not chunks:
            print(" -> Boş dosya, atlanıyor.")
            continue

        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f" -> Toplam {len(chunks)} parça ({total_batches} paket) işlenecek:")
        for batch_idx, i in enumerate(range(0, len(chunks), BATCH_SIZE), 1):
            batch = chunks[i:i + BATCH_SIZE]
            response = embedding_client.generate_embeddings(batch)
            for chunk, item in zip(batch, response.data):
                db.save_chunk(conn, source=file_name, content=chunk, embedding=item.embedding)
            percent = (batch_idx / total_batches) * 100
            print(f"\r   İlerleme: %{percent:.1f} ({batch_idx}/{total_batches} paket)", end="", flush=True)
        print()
        total_chunks += len(chunks)

    conn.close()
    embedding_model.unload()
    print(f"\n✓ İşlem tamam! Toplam {total_chunks} parça veritabanına aktarıldı.")


if __name__ == "__main__":
    main()
