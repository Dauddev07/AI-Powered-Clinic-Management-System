from app.rag.chunking import CHUNK_SIZE, chunk_text


def test_single_short_paragraph_stays_one_chunk():
    text = "The clinic opening hours are 9am to 5pm, Monday through Friday."

    chunks = chunk_text(text)

    assert chunks == [text]


def test_many_sections_each_land_in_their_own_chunk_not_merged():
    # Reproduces the dilution bug's shape: 12 departments joined the way
    # app.rag.extraction now joins paragraphs/table blocks ("\n\n"-separated).
    # Each one must become its own chunk so a query about one department's
    # doctors isn't diluted by the other eleven sharing its chunk.
    departments = [
        "The Cardiology department is staffed by Dr. Ahmed Farooq and Dr. Farhan Malik.",
        "The Dentistry department is staffed by Dr. Iqra Qureshi.",
        "The Dermatology department is staffed by Dr. Mariam Farooq and Dr. Shahid Sheikh.",
        "The ENT department is staffed by Dr. Iqra Raza and Dr. Waqas Farooq.",
        "The General Medicine department is staffed by Dr. Ali Raza and Dr. Farhan Rehman.",
        "The Gynecology department is staffed by Dr. Muhammad Qureshi and Dr. Rukhsana Hashmi.",
        "The Neurology department is staffed by Dr. Fatima Raza and Dr. Sana Qureshi.",
        "The Ophthalmology department is staffed by Dr. Farhan Mirza.",
        "The Orthopedics department is staffed by Dr. Junaid Mirza and Dr. Mahnoor Hashmi.",
        "The Pediatrics department is staffed by Dr. Ayesha Sheikh and Dr. Hina Raza.",
        "The Psychiatry department is staffed by Dr. Mariam Awan.",
        "The Pulmonology department is staffed by Dr. Babar Ali and Dr. Hassan Chaudhry.",
    ]
    text = "\n\n".join(departments)

    chunks = chunk_text(text)

    assert len(chunks) == len(departments)
    for department_sentence, chunk in zip(departments, chunks):
        assert chunk == department_sentence

    # No chunk mentions more than one department's doctors.
    cardiology_chunk = next(c for c in chunks if "Cardiology" in c)
    assert "Dermatology" not in cardiology_chunk
    assert "Pulmonology" not in cardiology_chunk


def test_oversized_single_section_falls_back_to_character_split():
    # One section (no blank-line boundary inside it) longer than CHUNK_SIZE has no
    # smaller natural boundary to respect, so it must still get split somehow rather
    # than shipped as one giant unretrievable chunk.
    text = "This is a single very long paragraph about the clinic. " * 40
    assert len(text) > CHUNK_SIZE

    chunks = chunk_text(text)

    assert len(chunks) > 1
    assert all(len(c) <= CHUNK_SIZE for c in chunks)


def test_table_block_with_internal_newlines_but_no_blank_line_stays_one_chunk():
    # app.rag.extraction emits a table as one block with single "\n" between rows
    # (no "\n\n" inside it) — that must still count as one section, not get split
    # row-by-row, since the table (e.g. a symptom-department reference) is one topic.
    table_block = "Department | Doctor | Shift\nCardiology | Dr. Ahmed Farooq | Mon 9am-1pm\nCardiology | Dr. Farhan Malik | Tue 2pm-6pm"

    chunks = chunk_text(table_block)

    assert chunks == [table_block]


def test_blank_lines_and_whitespace_only_sections_are_dropped():
    text = "First department paragraph.\n\n\n\n   \n\nSecond department paragraph."

    chunks = chunk_text(text)

    assert chunks == ["First department paragraph.", "Second department paragraph."]
