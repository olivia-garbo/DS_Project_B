import spacy, re, csv, os
from spacy.tokens import Span
from spacy.kb import KnowledgeBase, InMemoryLookupKB, get_candidates
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


def extend_person_entity(doc):
    # Create list to store new entities with extended boundaries
    Span.set_extension("person_title", getter=get_person_title, force=True)
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


def get_person_title(span):
    if span.label_ == "PERSON" and span.start != 0:
        prev_token = span.doc[span.start - 1]
        if prev_token.text in ("Dr", "Dr.", "Mr", "Mr.", "Ms", "Ms."):
            return prev_token.text


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


def get_entities(nlp, sent):
    ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""  # dependency tag of previous token in the sentence
    prv_tok_text = ""  # previous token in the sentence

    prefix = ""
    modifier = ""

    #############################################################

    for tok in nlp(sent):
        ## chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            ## chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

            ## chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            ## chunk 5
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
    #############################################################

    return [ent1.strip(), ent2.strip()]


def get_relation(nlp, sent):
    doc = nlp(sent)
    extend_person_entity(doc)
    # Matcher class object
    matcher = Matcher(nlp.vocab)
    # a is wife of b
    pattern_relation = [
        {"ENT_TYPE": "PERSON", "OP": "+"},  # First person entity
        {
            "LOWER": {"IN": ["is", "was", "'s", "are", "were", "am"]},
            "OP": "+",
        },  # Copula or possessive
        {"IS_ALPHA": True, "OP": "*"},  # Optional adjectives or determiners
        {
            "LOWER": {
                "IN": [
                    "wife",
                    "husband",
                    "brother",
                    "sister",
                    "father",
                    "mother",
                    "son",
                    "daughter",
                    "friend",
                    "cousin",
                    "uncle",
                    "aunt",
                    "nephew",
                    "niece",
                    "grandfather",
                    "grandmother",
                    "grandson",
                    "granddaughter",
                    "partner",
                    "lover",
                    "fiancé",
                    "fiancée",
                    "stepfather",
                    "stepmother",
                    "stepson",
                    "stepdaughter",
                    "stepbrother",
                    "stepsister",
                    "in-law",
                ]
            }
        },
        {"IS_PUNCT": True, "OP": "*"},
        {"LOWER": "of", "OP": "?"},
        {"IS_PUNCT": True, "OP": "*"},
        {"ENT_TYPE": "PERSON", "OP": "+"},  # Second person entity
    ]
    matcher.add("social_relation", [pattern_relation], greedy="LONGEST")
    # matcher.add("matching_1", [pattern])

    matches = matcher(doc)
    k = len(matches) - 1
    if k >= 0:
        span = doc[matches[k][1] : matches[k][2]]
        return span.text


def main():
    # load text from file
    nlp = spacy.load("en_core_web_lg")
    text = ""
    print("Loading text from file...")
    with open("resolved_book.txt", "r", encoding="utf-8") as file:
        text = file.read()
    # extend_person_entity(doc)
    # Load or build knowledge base
    mode = "sentence"  # Change this to "chapter", "paragraph", or "100token" as needed
    sentence = divide_text_by(nlp, text, by=mode)
    print(f"Total {len(sentence)} {mode}s")
    # sentence = ["Mr. Bennet is the sister of jane."]
    relations = [get_relation(nlp, sent) for sent in sentence]
    for rel in relations:
        if rel:
            print(rel)


if __name__ == "__main__":
    main()
