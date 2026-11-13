from retail_rag.documents import chunk_text, slugify


def test_slugify_handles_turkish_characters():
    assert slugify("İade süresi ve koşulları") == "iade-suresi-ve-kosullari"
    assert slugify("Ödeme seçenekleri ve taksit") == "odeme-secenekleri-ve-taksit"


def test_corpus_has_products_and_faq(corpus):
    sources = {d.source for d in corpus}
    assert sources == {"product", "faq"}
    ids = [d.id for d in corpus]
    assert len(ids) == len(set(ids)), "document ids must be unique"
    assert "faq-hediye-karti-0" in ids


def test_product_content_contains_price_and_attributes(corpus):
    boot = next(d for d in corpus if d.id == "p-001")
    assert "3299 TL" in boot.content
    assert boot.price == 3299.0 and boot.gender == "kadin"


def test_chunk_text_respects_limit_and_overlaps():
    paras = [f"paragraph {i} " + "x" * 80 for i in range(10)]
    chunks = chunk_text("\n\n".join(paras), max_chars=300, overlap_paragraphs=1)
    assert len(chunks) > 1
    assert all(len(c) <= 300 + 100 for c in chunks)
    # last paragraph of chunk n is the first of chunk n+1
    assert chunks[0].split("\n\n")[-1] == chunks[1].split("\n\n")[0]
