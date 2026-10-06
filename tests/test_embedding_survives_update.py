"""A block's vector and its graph edges survive an update — including the access count a plain get() records.
Until 2026-10-07, save_block used INSERT OR REPLACE; with foreign keys on, the implicit delete cascaded to
block_embeddings and edges, so reading a memory erased its vector (found by arrowOS's retrieval bench)."""
from memblock import MemBlock
from memblock.embeddings import EmbeddingProvider


class TinyEmbeddings(EmbeddingProvider):
    def embed(self, texts):
        return [[float(len(t)), 1.0, 0.5] for t in texts]

    @property
    def dimensions(self):
        return 3


def test_reading_and_updating_a_block_keeps_its_vector_and_edges():
    mem = MemBlock(storage="sqlite:///:memory:", embeddings=TinyEmbeddings())
    a = mem.store("I want to retire by 50.")
    b = mem.store("My retirement account is with a bank.")
    mem.link(a.id, b.id, relation="related_to")
    assert len(mem._storage.get_all_embeddings()) == 2
    mem.get(a.id)
    mem.get(a.id)
    mem.update(b.id, confidence=0.5)
    assert len(mem._storage.get_all_embeddings()) == 2
    assert [n.id for n in mem.neighbors(a.id)] == [b.id]
