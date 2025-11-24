from spacy.lang.en import English
from spacy.kb import KnowledgeBase, InMemoryLookupKB
from spacy.tokens import Span
import spacy
from spacy.matcher import Matcher


def remove_headers_footers(text):
    lines = text.split("\n")
    remaining_lines = lines[93:13336]
    text = "\n".join(remaining_lines)
    return text


def clean_name(name):
    removes = [" ", "\n", ".", '"', "'", "!"]
    if "--" in name:
        name = name.split("--")[0]
    for remove in removes:
        name = name.replace(remove, "")
    return name


print(clean_name("Wickham!--Your"))
# remove_headers_footers(
#     "Sample text with headers and footers\nHeader line\nMore text\nFooter line\nEnd of content",
#     stop_words=True,
# )
# def add_relationship(matcher, doc, i, matches):
#     match_id, start, end = matches[i]
#     entity = Span(doc, start, end, label="RELATIONSHIP")
#     doc.ents = list(doc.ents) + [entity]
#     print(f"Added relationship entity: {entity.text},{ entity.label_}")


# nlp = spacy.load("en_core_web_lg")
# relationship_patterns = [
#     [{"LOWER": "friend"}],
#     [{"LOWER": "couple"}],
#     [{"LOWER": "brother"}],
#     [{"LOWER": "married"}],
# ]

# # Create matcher
# matcher = Matcher(nlp.vocab)
# # Add patterns
# matcher.add("RELATIONSHIPS", relationship_patterns, on_match=add_relationship)
# matches = matcher(
#     nlp("Elizabeth Bennet and Mr. Darcy are a couple. and mary is a friend of Jane.")
# )
