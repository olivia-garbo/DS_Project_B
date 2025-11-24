import spacy
from spacy.tokens import Span
from spacy.matcher import Matcher


def main():
    # load text from file
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(
        "Adam is Eve's husband. Adam and Eve are the first humans created by God. Adam was created from dust, while Eve was created from Adam's rib. They lived in the Garden of Eden, where they were tempted by a serpent to eat the forbidden fruit. This act of disobedience led to their expulsion from the garden, and Adam has 1 billon dollars"
    )
    build_reliationships(doc, nlp)
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)


def add_relationship(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    entity = Span(doc, start, end, label="RELATIONSHIP")
    try:
        doc.ents = list(doc.ents) + [entity]
        # print(f"Added relationship entity: {entity.text},{ entity.label_}")
    except ValueError as e:
        print(f"Error adding entity: {e}")
        # Handle the case where the entity already exists or other issues


def build_reliationships(doc, nlp):
    relationship_patterns = [
        [{"lower": {"in": ["friend", "friends"]}}],  # matches both singular and plural
        [{"lower": "couple"}],
        [{"lower": "brother"}],
        [{"lower": "sister"}],
        [
            {"lower": {"in": ["daughter", "daughters"]}}
        ],  # matches both singular and plural
        [{"lower": {"in": ["son", "sons"]}}],
        [{"lower": "parent"}],
        [{"lower": "father"}],
        [{"lower": "mother"}],
        [{"lower": "wife"}],
        [{"lower": "husband"}],
    ]

    # create matcher
    matcher = Matcher(nlp.vocab)
    # add patterns
    matcher.add("relationships", relationship_patterns, on_match=add_relationship)
    matches = matcher(doc)


if __name__ == "__main__":
    main()
