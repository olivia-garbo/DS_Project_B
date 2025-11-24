import spacy


def remove_headers_footers(text, *opts):
    lines = text.split("\n")
    remaining_lines = lines[93:13336]
    text = "\n".join(remaining_lines)
    return text


def pre_process(remove_stop_words=False):
    print("Pre-processing text...")
    with open("42671.txt", "r", encoding="utf-8") as file:
        text = file.read()

    text = remove_headers_footers(text)
    # text = re.sub(
    #     r"^CHAPTER\s+[IVXLC\d]+\.", "", text, flags=re.IGNORECASE | re.MULTILINE
    # )
    if remove_stop_words:
        nlp = spacy.load("en_core_web_lg")
        doc = nlp(text)
        text = " ".join([token.text for token in doc if not token.is_stop])

    with open("clean_book.txt", "w", encoding="utf-8") as file:
        file.write(text)
    print("Pre-processing complete.")


if __name__ == "__main__":
    pre_process(remove_stop_words=False)
    print("Text pre-processed and saved to clean_book.txt.")
