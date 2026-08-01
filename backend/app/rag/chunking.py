from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# A blank line marks a genuine section boundary — app.rag.extraction joins each
# paragraph/table block with "\n\n", so this lines up with one section = one
# topic (e.g. one department's doctors, one table). Chunks never straddle this
# boundary even when merging two short sections would still fit under CHUNK_SIZE:
# a document listing many departments back-to-back must not have one department's
# content bleed into its neighbor's chunk and dilute its own retrieval signal.
_SECTION_SEPARATOR = "\n\n"


def _split_oversized_section(section: str) -> list[str]:
    """A single section that's still longer than CHUNK_SIZE has no smaller natural
    boundary left to respect, so it falls back to a plain character-count split."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )
    return splitter.split_text(section)


def chunk_text(text: str) -> list[str]:
    sections = [s.strip() for s in text.split(_SECTION_SEPARATOR) if s.strip()]

    chunks: list[str] = []
    for section in sections:
        if len(section) <= CHUNK_SIZE:
            chunks.append(section)
        else:
            chunks.extend(_split_oversized_section(section))
    return [c for c in chunks if c.strip()]
