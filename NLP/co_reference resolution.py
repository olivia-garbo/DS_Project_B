import re
import spacy
from spacy.tokens import Span
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_lg")


# ======================================================
# 0. 规则：强制合并 “Mr. Bennet” → PERSON 实体
# ======================================================
def merge_titles(doc):

    matcher = Matcher(nlp.vocab)

    TITLE = ["Mr", "Mr.", "Mrs", "Mrs.", "Miss", "Ms", "Lady", "Sir",
             "Colonel", "Capt", "Captain", "Lord", "Rev", "General"]

    pattern = [
        {"TEXT": {"IN": TITLE}},
        {"IS_ALPHA": True, "OP": "+"}
    ]

    matcher.add("TITLE_NAME", [pattern])
    matches = matcher(doc)

    new_spans = []

    for _, start, end in matches:
        # 创建新的 PERSON span
        span = Span(doc, start, end, label="PERSON")
        new_spans.append(span)

    # *** 关键：合并成一个列表后去重叠 ***
    all_spans = list(doc.ents) + new_spans
    all_spans = spacy.util.filter_spans(all_spans)   # <- ⭐ 必须在这里过滤

    doc.ents = all_spans
    return doc


# ======================================================
# 工具：生成 Bennet’s / Collins’
# ======================================================
def make_possessive(name):
    return name + "’" if name.endswith("s") else name + "’s"


# ======================================================
# 强替换核心逻辑：替换 he/him/his/she/her
# ======================================================
def patch_coref(sent, last_entity):

    if not last_entity:
        return sent

    name = last_entity.strip()

    # possessive
    poss = make_possessive(name)

    # 替换优先级：长的先替换
    sent = re.sub(r"\bhis\b", poss, sent, flags=re.I)
    sent = re.sub(r"\bher\b", poss, sent, flags=re.I)

    sent = re.sub(r"\bhe\b", name, sent, flags=re.I)
    sent = re.sub(r"\bhim\b", name, sent, flags=re.I)
    sent = re.sub(r"\bshe\b", name, sent, flags=re.I)

    return sent


# ======================================================
# 强替换主逻辑
# ======================================================
def strong_coref(sentences):

    memory = []
    output = []

    for sent in sentences:

        doc = merge_titles(nlp(sent))   # ⭐ 关键：强制合并 Mr. Bennet

        # 抓 PERSON 实体（title + surname）
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

        # 更新 memory
        for p in persons:
            if p not in memory:
                memory.append(p)

        memory = memory[-5:]   # 保留最近 5

        last_entity = memory[-1] if memory else None

        # 强替换 pronoun
        new_sent = patch_coref(sent, last_entity)
        output.append(new_sent)

    return "\n".join(output)


# ======================================================
# 主程序
# ======================================================
def run_coref():

    print("📘 Loading text…")
    with open("clean_book.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print("📘 Splitting sentences…")
    sentences = [s.text for s in nlp(text).sents]

    print("✨ Running PERSON-merged strong coreference…")
    resolved = strong_coref(sentences)

    print("💾 Saving resolved_book.txt…")
    with open("resolved_book.txt", "w", encoding="utf-8") as f:
        f.write(resolved)

    print("✅ DONE — resolved_book.txt updated!")


if __name__ == "__main__":
    run_coref()
