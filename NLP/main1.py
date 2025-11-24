import spacy, re, csv, os
from spacy.tokens import Span
from spacy.kb import KnowledgeBase, InMemoryLookupKB
from spacy.matcher import Matcher
import pandas as pd


def load_entities():
    entities_path = "characters.csv"
    names = dict()
    aliases = dict()
    with open(entities_path, "r", encoding="utf-8") as file:
        csvreader = csv.reader(file, delimiter=",")
        for row in csvreader:
            qid = row[0]
            name = row[1]
            alias = [
                alias.replace(" ", "").replace(".", "").replace("\n", "")
                for alias in row[2:]
            ]  # Remove empty aliases
            names[qid] = name.replace(" ", "").replace("\n", "").replace(".", "")
            aliases[qid] = alias

    return names, aliases


def clean_name(name):
    removes = [" ", "\n", ".", '"', "'", "!"]
    if "--" in name:
        name = name.split("--")[0]
    for remove in removes:
        name = name.replace(remove, "")
    return name


def get_person_title(span):
    if span.label_ == "PERSON" and span.start != 0:
        prev_token = span.doc[span.start - 1]
        if prev_token.text in ("Dr", "Dr.", "Mr", "Mr.", "Ms", "Ms."):
            return prev_token.text


def add_relationship(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    entity = Span(doc, start, end, label="RELATIONSHIP")
    try:
        doc.ents = list(doc.ents) + [entity]
        # print(f"Added relationship entity: {entity.text},{ entity.label_}")
    except ValueError as e:
        print(f"Error adding entity: {e}")
        # Handle the case where the entity already exists or other issues


def extend_person_entity(doc):
    # Create list to store new entities with extended boundaries
    Span.set_extension("person_title", getter=get_person_title)
    new_entities = []

    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent._.person_title:
            # Extend entity to include the title (one token before)
            extended_start = ent.start - 1
            extended_span = Span(doc, extended_start, ent.end, label="PERSON")
            new_entities.append(extended_span)
        else:
            # Keep all non-person entities unchanged
            new_entities.append(ent)

    # Update the document's entities with the extended boundaries
    doc.ents = new_entities


def build_knowledge_base(nlp):
    if os.path.exists("entity_link"):
        print("Loading existing knowledge base from disk...")
        kb = InMemoryLookupKB.from_disk("entity_link")
    else:
        kb = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=300)
        names_dict, aliases_dict = load_entities()
        entity_link = dict()
        # print(aliases_dict)
        for qid, name in names_dict.items():
            # Add entity to the knowledge base
            print(f"Adding entity: {qid} - {name}")
            kb.add_entity(entity=qid, entity_vector=nlp(name).vector, freq=342)
            # Add aliases for the entity
            for alias in aliases_dict[qid]:
                if alias:
                    kb.add_alias(entities=[qid], alias=alias, probabilities=[1])
        # print(kb.get_entity_strings())
        # print(kb.get_alias_strings())
        kb.to_disk("entity_link")

    return kb


def build_reliationships(doc, nlp):
    relationship_patterns = [
        [{"LOWER": {"IN": ["friend", "friends"]}}],  # Matches both singular and plural
        [{"LOWER": "couple"}],
        [{"LOWER": "brother"}],
        [{"LOWER": "sister"}],
        [
            {"LOWER": {"IN": ["daughter", "daughters"]}}
        ],  # Matches both singular and plural
        [{"LOWER": {"IN": ["son", "sons"]}}],
        [{"LOWER": "parent"}],
        [{"LOWER": "father"}],
        [{"LOWER": "mother"}],
        [{"LOWER": "wife"}],
        [{"LOWER": "husband"}],
    ]

    # Create matcher
    matcher = Matcher(nlp.vocab)
    # Add patterns
    matcher.add("RELATIONSHIPS", relationship_patterns, on_match=add_relationship)
    matches = matcher(doc)


def cluster_name_entities(doc, kb):
    names_dict, aliases_dict = load_entities()
    entity_link = dict()
    print("\nFinal entities:")
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Remove spaces from entity text
            clean_name = ent.text.replace(" ", "").replace("\n", "").replace(".", "")
            # If the cleaned name is in the knowledge base, get its ID
            for qid, name in names_dict.items():
                if name == clean_name:
                    kb_id = qid
                    if kb_id in entity_link and ent not in entity_link[kb_id]:
                        entity_link[kb_id].append(ent)
                    else:
                        entity_link[kb_id] = [ent]
                    break
                # get entity ID for alias
                elif kb.get_alias_candidates(clean_name):
                    kb_id = kb.get_alias_candidates(clean_name)[0].entity_
                    # print(f"Alias found for {clean_name}, using ID: {kb_id}")
                    if kb_id in entity_link and ent not in entity_link[kb_id]:
                        entity_link[kb_id].append(ent)
                    else:
                        entity_link[kb_id] = [ent]
                    break
                else:
                    kb_id = "N/A"
            print(f"PERSON: {clean_name}, KB ID: {kb_id}")


def divide_text_by(nlp, text, by=None):
    """
    Divide the document into chunks of specified size.
    """
    if by == "chapter":
        # Divide by chapter
        chapters = re.split(
            r"^CHAPTER\s+[IVXLC\d]+\.", text, flags=re.IGNORECASE | re.MULTILINE
        )
        chapters.pop(0)
        return chapters
    elif by == "paragraph":
        paragraphs = text.split("\n")
        paragraphs = [
            p.strip()
            for p in paragraphs
            if p.strip() and not re.match(r"^CHAPTER\s+", p, re.IGNORECASE)
        ]
        return paragraphs
    elif by == "sentence":
        if nlp is None:
            raise ValueError("nlp object is required for sentence-based chunking")

        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences

    elif by == "100token":
        if nlp is None:
            raise ValueError("nlp object is required for entity-based chunking")
        doc = nlp(text)
        chunks = []
        current_chunk = ""
        entity_count = 0
        # Get all entities with their positions
        entities = list(doc.ents)

        if not entities:
            return [text]  # Return original text if no entities found

        # Split text into sentences for better chunking
        sentences = list(doc.sents)
        current_chunk_sentences = []
        for sent in sentences:
            # Count entities in this sentence
            sent_entities = [
                ent
                for ent in entities
                if ent.start >= sent.start and ent.end <= sent.end
            ]

            # If adding this sentence would exceed 100 entities, save current chunk
            if entity_count + len(sent_entities) > 100 and current_chunk_sentences:
                chunks.append(" ".join([s.text for s in current_chunk_sentences]))
                current_chunk_sentences = [sent]
                entity_count = len(sent_entities)
            else:
                current_chunk_sentences.append(sent)
                entity_count += len(sent_entities)

        # Add the last chunk if it has content
        if current_chunk_sentences:
            chunks.append(" ".join([s.text for s in current_chunk_sentences]))
        print(f"Divided text into {len(chunks)} chunks with ~100 entities each")
        return chunks


def chapter_parse_relations(chunks, nlp) -> list:
    relationship_buffer = []
    relationships = []
    for chunk in chunks:
        doc = nlp(chunk)
        build_reliationships(doc, nlp)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                if len(relationship_buffer) == 0:
                    relationship_buffer.append(ent)
                elif relationship_buffer[-1].label_ == "RELATIONSHIP":
                    relationships.append(
                        (relationship_buffer.pop(), relationship_buffer.pop(), ent)
                    )
                    relationship_buffer.clear()
                elif relationship_buffer[-1].label_ == "PERSON":
                    relationship_buffer.clear()
                    relationship_buffer.append(ent)
            elif ent.label_ == "RELATIONSHIP":
                if (
                    len(relationship_buffer) > 0
                    and relationship_buffer[-1].label_ == "PERSON"
                ):
                    relationship_buffer.append(ent)
        relationship_buffer.clear()
    # print(relationships)
    return relationships


def consolidate_relationships_entities(relationships, kb, mode):
    """
    Consolidate relationships and entities into the knowledge base.
    """
    print(kb.get_alias_strings())

    def convert_name_to_kbid(ent):
        """
        Get alias candidates for a given name from the knowledge base.
        """
        if ent._.person_title:
            name = f"{ent._.person_title }{ent.text}"
        else:
            name = ent.text
        name = clean_name(name)
        # print(f"Getting alias candidates for: {name}")
        candidates = kb.get_alias_candidates(name)
        if candidates:
            return candidates[0].entity_
        else:
            candidates = kb.get_candidates(ent)
            return candidates[0].entity_ if candidates else "N/A"

    df = pd.DataFrame(relationships, columns=["Relationship", "Entity1", "Entity2"])
    df["Entity1_ID"] = df["Entity1"].apply(convert_name_to_kbid)
    df["Entity2_ID"] = df["Entity2"].apply(convert_name_to_kbid)
    print(df.head(100))
    df.to_csv(f"conslidated_relationships.csv", index=False)
    print("Consolidated relationships and entities into consolidated_relationships.csv")


def load_knowledge_base(nlp):
    """
    Load the knowledge base from disk.
    """
    kb = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=300)
    if os.path.exists("entity_link"):
        print("Loading existing knowledge base from disk...")
        kb.from_disk("./entity_link")  # Load from the .kb file
        return kb
    else:
        print("No knowledge base found on disk.")
        return None


def main():
    # load text from file
    nlp = spacy.load("en_core_web_lg")
    text = ""
    print("Loading text from file...")
    with open("resolved_book.txt", "r", encoding="utf-8") as file:
        text = file.read()
    doc = nlp(text)
    extend_person_entity(doc)
    # Load or build knowledge base
    if not os.path.exists("entity_link"):
        build_knowledge_base(nlp)
    kb = load_knowledge_base(nlp)
    mode = "sentence"  # Change this to "chapter", "paragraph", or "100token" as needed
    chapters = divide_text_by(nlp, text, by=mode)
    relationships = chapter_parse_relations(chapters, nlp)
    if kb:
        consolidate_relationships_entities(relationships, kb, mode=mode)


if __name__ == "__main__":
    main()
