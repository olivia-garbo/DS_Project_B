import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
import os
from itertools import chain


def load_names():
    """Load names from characters_updated.csv and return QID→Name dict."""
    try:
        df = pd.read_csv("characters_updated.csv", encoding="utf-8")
        if "QID" not in df.columns or "Name" not in df.columns:
            df.columns = ["QID", "Name", "Aliases"]
        df["QID"] = df["QID"].astype(str).str.strip()
        df["Name"] = df["Name"].astype(str).str.strip()
        return dict(zip(df["QID"], df["Name"]))
    except FileNotFoundError:
        print("❌ Error: 'characters_updated.csv' not found.")
        return {}


# ======================================================
#  PROCESS DATA
# ======================================================
def process_data():
    input_path = "consolidated_relationships.csv"  # ✅ 修正文件名
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found in current directory.")
        return

    df = pd.read_csv(input_path, encoding="utf-8")
    df = df.dropna(subset=["Entity1_ID", "Entity2_ID", "Relationship"])

    df["Entity1_ID"] = df["Entity1_ID"].astype(str).str.strip()
    df["Entity2_ID"] = df["Entity2_ID"].astype(str).str.strip()
    df = df[df["Entity1_ID"] != df["Entity2_ID"]]

    # 🔹 标准化 Relationship
    def standardize_relationship(relationship):
        if not isinstance(relationship, str):
            return relationship
        relationship = relationship.lower().strip()
        mapping = {
            "friends": "friend",
            "daughters": "daughter",
            "sons": "son",
            "brothers": "brother",
            "sisters": "sister",
            "parents": "parent",
            "couples": "couple",
            "wives": "wife",
            "husbands": "husband",
            "fathers": "father",
            "mothers": "mother",
        }
        return mapping.get(relationship, relationship)

    df["Relationship"] = df["Relationship"].apply(standardize_relationship)

    # 🔹 创建无向 pair
    df["sorted_pair"] = df.apply(
        lambda r: tuple(sorted([r["Entity1_ID"], r["Entity2_ID"]])), axis=1
    )

    # 🔹 各类计数
    rel_type_counts = (
        df.groupby(["sorted_pair", "Relationship"])
        .size()
        .reset_index(name="relationship_type_count")
    )
    total_counts = (
        df.groupby("sorted_pair")
        .size()
        .reset_index(name="total_relationship_count")
    )
    unique_counts = (
        df.groupby("sorted_pair")["Relationship"]
        .nunique()
        .reset_index(name="unique_relationship_types")
    )

    # 🔹 合并统计
    df_final = rel_type_counts.merge(total_counts, on="sorted_pair", how="left")
    df_final = df_final.merge(unique_counts, on="sorted_pair", how="left")
    #df_final.drop_duplicates(inplace=True)

    # 🔹 拆分 Entity ID
    df_final[["Entity1_ID", "Entity2_ID"]] = df_final["sorted_pair"].apply(
        lambda x: pd.Series(eval(str(x)))
    )

    # 🔹 映射人名
    name_dict = load_names()
    df_final["Entity1"] = df_final["Entity1_ID"].map(name_dict)
    df_final["Entity2"] = df_final["Entity2_ID"].map(name_dict)

    # 🔹 保存结果
    os.makedirs("results", exist_ok=True)
    counts_path = "results/relationships_with_counts.csv"
    df_final.to_csv(counts_path, index=False, encoding="utf-8")

    # 🔹 生成 pivot 汇总
    pivot = (
        df.pivot_table(
            index="sorted_pair",
            columns="Relationship",
            values="Entity1_ID",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )
    pivot_path = "results/relationship_pivot_summary.csv"
    pivot.to_csv(pivot_path, index=False, encoding="utf-8")

    print(f"✅ Saved: {counts_path}")
    print(f"✅ Saved: {pivot_path}")
    print(f"✅ Total pairs processed: {len(df_final)}")


# ======================================================
#  DRAW GRAPH
# ======================================================
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
import os
from itertools import chain

def draw_graph():
    pivot_path = "results/relationship_pivot_summary.csv"
    if not os.path.exists(pivot_path):
        print("⚠️ Please run process_data() first.")
        return

    # === 1️⃣ 读取透视表 ===
    df = pd.read_csv(pivot_path)
    pairs = df["sorted_pair"].apply(eval)
    all_ids = list(set(chain.from_iterable(pairs)))

    # === 2️⃣ 构建图节点 ===
    g = ig.Graph(directed=False)
    g.add_vertices(all_ids)

    # 加载角色名
    name_dict = load_names()
    g.vs["label"] = [name_dict.get(i, i) for i in all_ids]

    # === 3️⃣ 节点大小 ∝ 出现频率 ===
    char_path = "characters.csv"
    mention_dict = {}
    if os.path.exists(char_path):
        chars_df = pd.read_csv(char_path, header=None)
        if chars_df.shape[1] > 2:
            mention_dict = dict(zip(
                chars_df.iloc[:, 0],
                pd.to_numeric(chars_df.iloc[:, 2], errors="coerce").fillna(1).astype(int)
            ))
        else:
            mention_dict = {qid: 1 for qid in chars_df.iloc[:, 0]}
    else:
        mention_dict = {qid: 1 for qid in all_ids}

    mentions = [mention_dict.get(v, 1) for v in all_ids]
    mentions = [int(m) if str(m).isdigit() else 1 for m in mentions]
    min_size, max_size = 40, 120
    if max(mentions) > 0:
        v_sizes = [min_size + (m / max(mentions)) * (max_size - min_size) for m in mentions]
    else:
        v_sizes = [min_size for _ in mentions]
    g.vs["size"] = v_sizes

    # === 4️⃣ 计算每条边的主要关系 + 强度 ===
    relation_cols = df.columns[1:]
    df[relation_cols] = df[relation_cols].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0).astype(int)
    df["main_relationship"] = df[relation_cols].idxmax(axis=1)
    df["relation_strength"] = df[relation_cols].max(axis=1)

    # === 5️⃣ 添加边（带关系+权重） ===
    for _, row in df.iterrows():
        e1, e2 = eval(row["sorted_pair"])
        rel = row["main_relationship"]
        weight = int(row["relation_strength"]) if not pd.isna(row["relation_strength"]) else 1
        if e1 in all_ids and e2 in all_ids:
            g.add_edge(e1, e2, relationship=rel, weight=weight, use_vids=False)

    # === 6️⃣ 边宽度 ∝ 关系强度 ===
    weights = g.es["weight"]
    min_w, max_w = 0.8, 5
    if max(weights) > 0:
        e_widths = [
            min_w + (w / max(weights)) * (max_w - min_w) for w in weights
        ]
    else:
        e_widths = [min_w for _ in weights]

    # === 7️⃣ 绘图 ===
    fig, ax = plt.subplots(figsize=(12, 12))
    ig.plot(
        g,
        target=ax,
        layout=g.layout("fruchterman_reingold"),
        vertex_size=g.vs["size"],
        vertex_color="lightblue",
        vertex_label=g.vs["label"],
        vertex_label_size=8,
        edge_label=[rel for rel in g.es["relationship"]],
        edge_label_size=6,
        edge_color="gray",
        edge_width=e_widths,
    )
    plt.title("Character Relationship Network (Weighted by Mentions & Frequency)")
    plt.show()

    # === 8️⃣ 保存图结构 ===
    g.write_gml("results/character_relationships_weighted.gml")
    print("✅ Weighted graph saved as results/character_relationships_weighted.gml")


# ======================================================
if __name__ == "__main__":
    process_data()
    draw_graph()

