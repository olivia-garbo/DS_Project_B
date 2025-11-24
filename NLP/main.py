import spacy
import pandas as pd
import networkx as nx

import re, itertools
import matplotlib.pyplot as plt
import igraph as ig


def remove_headers_footers(text):
    lines = text.split("\n")

    remaining_lines = lines[93:13336]
    text = "\n".join(remaining_lines)
    return text


def show_result(character_relations_dictionary, characters):
    character_names = [char[0] for char in characters]
    name_to_index = {name: i for i, name in enumerate(character_names)}
    n_vertices = len(characters)
    # Convert character name pairs to index pairs
    edges = []
    weights = []
    for char_pair in character_relations_dictionary.keys():
        char1, char2 = char_pair
        if char1 in name_to_index and char2 in name_to_index:
            edges.append((name_to_index[char1], name_to_index[char2]))
            weights.append(character_relations_dictionary[char_pair])

    g = ig.Graph(n_vertices, edges=edges, directed=False)
    # Add character names as vertex labels
    g.vs["name"] = character_names
    g.es["weight"] = weights
    print(f"Edges (as indices): {edges}")
    print(f"Weights: {weights}")
    print(f"Character names: {character_names}")
    fig, ax = plt.subplots(figsize=(8, 8))
    ig.plot(
        g,
        target=ax,
        vertex_size=40,
        vertex_color=["steelblue"],
        vertex_frame_width=4.0,
        vertex_frame_color="white",
        vertex_label=g.vs["name"],
        vertex_label_size=8.0,
        # edge_width=[2 if married else 1 for married in g.es["married"]],
        # show important relations if weight > 10
        edge_color=[
            "#7142cf" if w > 10 else "#AAA" for w in g.es["weight"]
        ],  # Use weights for edge color
    )

    plt.show()


def main():
    # load text from file
    # r=read
    with open("42671.txt", "r", encoding="utf-8") as file:
        text = file.read()

    text = remove_headers_footers(text)
    text = text.replace(".", "")
    # Split chapters using regex that starts with CHAPTER
    # ^ means start of line, \s+ means one or more whitespace characters
    # This regex captures chapter headings like "CHAPTER I.", "CHAPTER II.", etc
    chapters = re.split(
        r"^CHAPTER\s+[IVXLC\d]+", text, flags=re.IGNORECASE | re.MULTILINE
    )

    # remove . for Mr./Mrs. etc to match the word in character file.
    # remove the first item which is empty
    chapters.pop(0)

    print(f"Number of chapters found: {len(chapters)}")
    # if len(chapters) > 1:
    #     print(chapters[0][:300])
    # nlp=spacy.load("en_core_web_sm")

    # doc=nlp(text)
    # doc.

    #    Save the processed text to a new file
    # with open("clean_book.txt", "w", encoding="utf-8") as file:
    #     file.write(text)

    #    Split book into paragraphs. Remove empty lines and CHAPTER headings
    paragraphs = text.split("\n")
    paragraphs = [
        p.strip()
        for p in paragraphs
        if p.strip() and not re.match(r"^CHAPTER\s+", p, re.IGNORECASE)
    ]

    # print(
    #     f"This book has {len(paragraphs)} paragraphs"
    # )  # Print total number of paragraphs

    # print(paragraphs[:5])  # Print first 5 paragraphs for verification
    with open("characters.txt", "r", encoding="utf-8") as file:
        characters = file.read().splitlines()
        characters = [
            c.split(",") for c in characters if c.strip() and not c.startswith("#")
        ]

    # save edges
    character_relations_dictionary = dict()
    for chapter in chapters:
        appears = []
        for character in characters:
            for name in character:
                if name in chapter:
                    appears.append(character[0])
                    break
        # print(appears)
        relationships = itertools.combinations(sorted(appears), 2)
        for relationship in relationships:
            # print(relationship)
            if relationship in character_relations_dictionary:
                character_relations_dictionary[relationship] += 1
            else:
                character_relations_dictionary[relationship] = 1

    show_result(character_relations_dictionary, characters)


if __name__ == "__main__":
    main()
