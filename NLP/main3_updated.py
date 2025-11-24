import spacy, re, csv, os
from spacy.tokens import Span
from spacy.kb import InMemoryLookupKB
from spacy.matcher import Matcher
from spacy.util import filter_spans
import pandas as pd


# ========== 1. 加载人物实体 ==========
def load_entities():
    entities_path = "characters_updated.csv"
    names, aliases = {}, {}
    with open(entities_path, "r", encoding="utf-8") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            if not row:
                continue
            qid, name = row[0], row[1]
            alias = [a.strip().replace(" ", "").replace(".", "") for a in row[2:] if a.strip()]
            names[qid] = name.strip().replace(" ", "").replace(".", "")
            aliases[qid] = alias
    return names, aliases


# ========== 2. 清洗名字 ==========
def clean_name(name):
    for ch in [" ", "\n", ".", '"', "'", "!"]:
        name = name.replace(ch, "")
    if "--" in name:
        name = name.split("--")[0]
    return name


# ========== 3. 称谓扩展 ==========
def get_person_title(span):
    titles = ["Mr", "Mr.", "Mrs", "Mrs.", "Miss", "Ms", "Lady", "Sir",
              "Colonel", "Capt", "Captain", "Lord", "Rev", "Rev.", "General", "Gen."]
    start = span.start
    if start > 0:
        token_before = span.doc[start - 1]
        if token_before.text in titles:
            return token_before.text
    return None


def extend_person_entity(doc):
    new_ents = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            title = get_person_title(ent)
            if title:
                new_ent = Span(doc, ent.start - 1, ent.end, label="PERSON")
                new_ent._.set("person_title", title)
                new_ents.append(new_ent)
            else:
                ent._.set("person_title", "")
                new_ents.append(ent)
        else:
            new_ents.append(ent)
    new_ents = filter_spans(new_ents)
    doc.ents = new_ents
    return doc


Span.set_extension("person_title", default="")


# ========== 4. 关系词标注 ==========
def build_reliationships(doc, nlp):
    matcher = Matcher(nlp.vocab)
    relationship_terms = [
        "friend", "friends", "brother", "sister", "daughter", "daughters",
        "son", "sons", "father", "mother", "wife", "husband", "aunt", "uncle",
        "niece", "nephew", "cousin", "in-law", "fiancé", "fiancée"
    ]
    patterns = [[{"LOWER": rel}] for rel in relationship_terms]
    matcher.add("RELATIONSHIP", patterns)
    matches = matcher(doc)
    new_ents = list(doc.ents)
    for _, start, end in matches:
        span = Span(doc, start, end, label="RELATIONSHIP")
        new_ents.append(span)
    new_ents = filter_spans(new_ents)
    doc.ents = new_ents
    return doc


# ========== 5. 构建知识库 ==========
def build_knowledge_base(nlp):
    names, aliases = load_entities()
    kb = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=300)
    for qid, name in names.items():
        kb.add_entity(entity=qid, freq=1, entity_vector=nlp(name).vector)
        for alias in aliases[qid]:
            kb.add_alias(alias, entities=[qid], probabilities=[1.0])
        kb.add_alias(name, entities=[qid], probabilities=[1.0])
    os.makedirs("entity_link", exist_ok=True)
    kb.to_disk("entity_link/kb")
    return kb


def load_knowledge_base(nlp):
    """
    ✅ 加载 characters_updated.csv 为 Knowledge Base（支持 Name + Aliases）
    ✅ 自动清洗空格、句号、大小写
    ✅ 多别名支持: 用分号 / 逗号 / 制表符分割
    """

    kb = {}
    filepath = "characters_updated.csv"

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ {filepath} not found!")

    with open(filepath, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header_skipped = False

        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            if not header_skipped:
                header_skipped = True
                continue

            qid = row[0].strip()
            name = row[1].strip()

            # 🔹 支持多种分隔符，并去掉引号
            aliases = []
            if len(row) > 2 and row[2].strip():
                alias_str = row[2].replace('"', '').replace("'", "")
                aliases = [
                    a.strip()
                    for a in re.split(r"[;,\t]", alias_str)
                    if a.strip()
                ]

            kb[qid] = {"name": name, "aliases": aliases}

    print(f"✅ Loaded {len(kb)} characters from knowledge base.")
    return kb



# ========== 6. 文本分块 ==========
def divide_text_by(nlp, text, by="sentence"):
    if by == "chapter":
        return re.split(r"CHAPTER [IVXLC]+", text)
    elif by == "paragraph":
        return [p for p in text.split("\n") if len(p.strip()) > 30]
    elif by == "sentence":
        return [sent.text for sent in nlp(text).sents]
    elif by == "100token":
        sents = [sent.text for sent in nlp(text).sents]
        chunks, chunk, token_count = [], [], 0
        for sent in sents:
            token_count += len(sent.split())
            chunk.append(sent)
            if token_count >= 100:
                chunks.append(" ".join(chunk))
                chunk, token_count = [], 0
        if chunk:
            chunks.append(" ".join(chunk))
        return chunks
    else:
        return [text]


# ========== 7. main1：顺序匹配 ==========
def extract_relationships_bidirectional(doc, filter_pronoun=False):
    """
    基于实体顺序的关系抽取（改进版）
    - 逻辑核心：PERSON — RELATIONSHIP — PERSON 顺序匹配
    - 不去掉代词 (her, his, their)；保留以捕捉 "her friend Charlotte" 等句式
    - 自动标准化关系词为小写
    """

    # 🔹 定义关键词
    REL_WORDS = {
        "father", "mother", "brother", "sister", "wife", "husband",
        "son", "daughter", "parent", "aunt", "uncle", "cousin",
        "nephew", "niece", "in-law", "fiancé", "fiancée",
        "friend", "lover", "partner", "companion", "relative", "family", "couple"
    }

    # 🔹 定义代词集合（可供后续分析使用）
    PRONOUNS = {"his", "her", "their", "my", "your", "our", "its", "him", "me", "them", "you"}

    # 🔹 只保留 PERSON / RELATIONSHIP 实体
    ents = [ent for ent in doc.ents if ent.label_ in ["PERSON", "RELATIONSHIP"]]

    relationships = []

    # 🔹 遍历实体，寻找关系词居中的模式
    for i, ent in enumerate(ents):
        if ent.label_ == "RELATIONSHIP" and ent.text.lower() in REL_WORDS:

            # 找到左侧第一个 PERSON（保留 pronoun）
            left_person = next(
                (e for e in reversed(ents[:i]) if e.label_ == "PERSON"),
                None
            )

            # 找到右侧第一个 PERSON（保留 pronoun）
            right_person = next(
                (e for e in ents[i + 1:] if e.label_ == "PERSON"),
                None
            )

            if left_person and right_person:
                relationships.append((
                    ent.text.lower(),
                    left_person,
                    right_person
                ))

    return relationships




# ========== 8. main2：依存句法匹配 ==========
def extract_dependency_relations(doc):
    REL_WORDS = {
        "father", "mother", "brother", "sister", "wife", "husband",
        "son", "daughter", "parent", "aunt", "uncle", "cousin",
        "nephew", "niece", "in-law", "fiancé", "fiancée", "friend"
    }
    PRONOUNS = {"his", "her", "their", "my", "your", "our", "its", "him", "me", "them", "you"}
    relationships = []

    for token in doc:

        # ============================================
        # B. copular: "Jane is Elizabeth’s sister"
        # ============================================
        if token.dep_ == "attr" and token.lemma_.lower() in REL_WORDS:
            subject = [w for w in token.head.lefts if w.dep_ == "nsubj" and w.ent_type_=="PERSON"]
            possessor = [w for w in token.children if w.dep_ == "poss" and w.ent_type_=="PERSON"]

            if subject and possessor:
                relationships.append((token.text, str(subject[0]), possessor[0].text))

        # ============================================
        # C. of-phrase: "sister of Elizabeth"
        # ============================================
        if token.lemma_.lower() in REL_WORDS:
            for child in token.children:
                if child.dep_ == "prep" and child.text.lower() == "of":
                    persons = [c for c in child.children if c.ent_type_=="PERSON"]
                    if persons:
                        e2 = persons[0]
                        e1 = token.head if token.head.ent_type_=="PERSON" else None
                        if e1:
                            relationships.append((token.text, str(e1), e2.text))

        # ============================================
        # D. apposition: "Mr Bennet, father of Jane"
        # ============================================
        if token.dep_ == "appos" and token.lemma_.lower() in REL_WORDS:
            head = token.head
            if head.ent_type_=="PERSON":
                for child in token.children:
                    if child.dep_=="prep" and child.text.lower()=="of":
                        persons = [c for c in child.children if c.ent_type_=="PERSON"]
                        for e2 in persons:
                            relationships.append((token.text, head.text, e2.text))

    

        # ============================================
        # F. NP-modifier: "Elizabeth’s sister Jane"
        # ============================================
        if token.lemma_.lower() in REL_WORDS:
            poss = [c for c in token.children if c.dep_=="poss" and c.ent_type_=="PERSON"]
            appos = [c for c in token.children if c.dep_=="appos" and c.ent_type_=="PERSON"]
            if poss and appos:
                relationships.append((token.text, appos[0].text, poss[0].text))

    return relationships


# ========== 9. main1 sentence ONLY ==========
def chapter_parse_relations(sentence_chunks, nlp):
    all_relationships = []

    for i, chunk in enumerate(sentence_chunks):
        doc = nlp(chunk)
        extend_person_entity(doc)
        build_reliationships(doc, nlp)

        rels1 = extract_relationships_bidirectional(doc)
        for rel, e1, e2 in rels1:
            all_relationships.append((rel, e1, e2))

        if (i + 1) % 300 == 0:
            print(f"🟦 main1 processed {i+1}/{len(sentence_chunks)} sentences")

    return all_relationships




# ========== 10. KB映射 & 输出（终极稳定版） ==========
def consolidate_relationships_entities(relationships, kb, default_mode="sentence"):
    """
    兼容 3 / 4 / 5 元组的通用 consolidate 函数。
    自动：
    - 清洗实体
    - 匹配 KB
    - 输出标准化名字
    - 输出 Mode / Source
    """

    VALID_REL_WORDS = {
        "father", "mother", "brother", "sister", "wife", "husband",
        "son", "daughter", "parent", "aunt", "uncle", "cousin",
        "nephew", "niece", "friend", "lover", "partner", "companion",
        "relative", "family", "couple"
    }

    PRONOUNS = {"his", "her", "their", "my", "your", "our", "its",
                "him", "me", "them", "you"}

    # ========== 清洗展示姓名 ==========
    def clean_name_display(name):
        for ch in [".", '"', "'", "!", "\n"]:
            name = name.replace(ch, "")
        return name.strip()

    # ========== KB 匹配函数 ==========
    def match_to_kb(ent):
        """
        输入 entity(span or str)
        输出 (QID, 标准KB名字)
        """

        raw = ent.text if hasattr(ent, "text") else str(ent)
        cleaned = re.sub(r"[\s\u00A0\u200B]+", "", raw) \
                    .replace(".", "").replace("'", "") \
                    .replace('"', "").lower().strip()

        for qid, entry in kb.items():
            name_clean = entry["name"].replace(" ", "").replace(".", "").lower()
            alias_clean_list = [a.replace(" ", "").replace(".", "").lower()
                                for a in entry.get("aliases", [])]

            # 精确匹配
            if cleaned == name_clean or cleaned in alias_clean_list:
                return qid, entry["name"]

            # 模糊包含匹配
            if cleaned in name_clean or any(cleaned in a for a in alias_clean_list):
                return qid, entry["name"]

        return "N/A", raw  # 匹配不到时使用原文本

    # ========== 整合输出 ==========
    rows = []

    for item in relationships:

        # --- 自动识别 3/4/5 元组 ---
        if len(item) == 3:
            rel, ent1, ent2 = item
            mode_used, source = default_mode, "unknown"

        elif len(item) == 4:
            rel, ent1, ent2, source = item
            mode_used = default_mode

        elif len(item) == 5:
            rel, ent1, ent2, mode_used, source = item

        else:
            print("⚠️ Unexpected relationship tuple:", item)
            continue

        # --- 过滤不在词典中的关系 ---
        if rel.lower() not in VALID_REL_WORDS:
            continue

        # --- 提取文本 ---
        ent1_text = ent1.text if hasattr(ent1, "text") else str(ent1)
        ent2_text = ent2.text if hasattr(ent2, "text") else str(ent2)

        # --- 过滤代词 ---
        if ent1_text.lower() in PRONOUNS or ent2_text.lower() in PRONOUNS:
            continue

        # --- KB 映射 ---
        ent1_id, ent1_std = match_to_kb(ent1)
        ent2_id, ent2_std = match_to_kb(ent2)

        # --- 写入行 ---
        rows.append({
            "Relationship": rel.lower(),
            "Entity1": ent1_std,
            "Entity2": ent2_std,
            "Entity1_ID": ent1_id,
            "Entity2_ID": ent2_id,
            "Mode": mode_used,
            "Source": source
        })

    # ========== 保存 CSV ==========
    df = pd.DataFrame(rows)
    df.to_csv("consolidated_relationships.csv", index=False, encoding="utf-8")

    print(f"🎉 Saved {len(df)} relationships → consolidated_relationships.csv")



# ========== 11. 主程序 ==========
def main():
    print("✅ Loading spaCy model...")
    nlp = spacy.load("en_core_web_lg")

    print("===================================================")
    print("📘 Loading original text (for main1 sequential)...")
    with open("clean_book.txt", "r", encoding="utf-8") as f:
        text_original = f.read()

    print("📗 Loading coreference-resolved text (for main2 dep)...")
    with open("resolved_book.txt", "r", encoding="utf-8") as f:
        text_resolved = f.read()

    print("===================================================")
    print("🔎 Loading Knowledge Base...")
    kb = load_knowledge_base(nlp)

    # ============================
    # main1: sequential, raw text
    # ============================
    print("===================================================")
    print("🚀 Extracting main1 relationships from ORIGINAL text (sentence mode)...")
    chunks_main1 = divide_text_by(nlp, text_original, by="sentence")
    print(f"➡️ main1: {len(chunks_main1)} sentences")

    rels_main1 = chapter_parse_relations(chunks_main1, nlp)
    rels_main1_labeled = [(r[0], r[1], r[2], "sentence", "main1") for r in rels_main1]
    print(f"✔ main1 extracted {len(rels_main1_labeled)} relations")


    # ============================
    # main2: dependency, COREF text
    # ============================
    print("===================================================")
    print("🧠 Extracting main2 relationships from COREFERENCE text (100-token)...")
    chunks_main2 = divide_text_by(nlp, text_resolved, by="100token")
    print(f"➡️ main2: {len(chunks_main2)} chunks")

    rels_main2 = []
    for i, chunk in enumerate(chunks_main2):
        doc = nlp(chunk)
        extend_person_entity(doc)
        build_reliationships(doc, nlp)
        rels = extract_dependency_relations(doc)
        rels_main2.extend([(r[0], r[1], r[2], "100token", "main2") for r in rels])
        if (i+1) % 10 == 0:
            print(f"   ✓ processed {i+1}/{len(chunks_main2)} chunks")

    print(f"✔ main2 extracted {len(rels_main2)} relations")


    # ============================
    # Merge results
    # ============================
    all_relationships = rels_main1_labeled + rels_main2
    print("===================================================")
    print(f"🔗 TOTAL merged relationships: {len(all_relationships)}")

    # ============================
    # consolidate
    # ============================
    print("===================================================")
    print("📦 Consolidating results with KB...")
    consolidate_relationships_entities(all_relationships, kb, default_mode="mixed")

    print("🎉 DONE!")


if __name__ == "__main__":
    main()