import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
import os


def load_names():
    """Load names from a CSV file and return a dictionary mapping IDs to names."""
    try:
        df = pd.read_csv("characters.csv", header=None)
        return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))
    except FileNotFoundError:
        print("Error: 'names.csv' not found.")
        return {}


# region process_data
def process_data():
    # Load the CSV file into a DataFrame
    df = pd.read_csv("conslidated_relationships.csv")

    # First, drop rows with NaN values in Entity columns
    df = df.dropna(subset=["Entity1_ID", "Entity2_ID"])

    # Convert Entity IDs to strings to ensure consistent data types
    df["Entity1_ID"] = df["Entity1_ID"].astype(str)
    df["Entity2_ID"] = df["Entity2_ID"].astype(str)

    # Drop rows where entity1_id and entity2_id are the same
    df = df[df["Entity1_ID"] != df["Entity2_ID"]]

    # Standardize relationship types (combine singular and plural forms)
    def standardize_relationship(relationship):
        """Standardize relationship types to singular forms"""
        relationship = relationship.lower().strip()

        # Define mapping of plural to singular forms
        standardization_map = {
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

        return standardization_map.get(relationship, relationship)

    # Apply standardization
    df["Relationship"] = df["Relationship"].apply(standardize_relationship)

    # Create a sorted tuple of the two IDs to identify bidirectional pairs
    df["sorted_pair"] = df.apply(
        lambda row: tuple(sorted([row["Entity1_ID"], row["Entity2_ID"]])), axis=1
    )

    # Count each relationship type for each sorted pair
    relationship_type_counts = (
        df.groupby(["sorted_pair", "Relationship"])
        .size()
        .reset_index(name="relationship_type_count")
    )

    # Get the first occurrence of each unique pair-relationship combination
    df_unique = df.drop_duplicates(subset=["sorted_pair", "Relationship"], keep="first")

    # Merge the relationship type counts back to the dataframe
    df_final = df_unique.merge(
        relationship_type_counts, on=["sorted_pair", "Relationship"], how="left"
    )

    # Count total relationships per pair
    total_counts = (
        df.groupby("sorted_pair").size().reset_index(name="total_relationship_count")
    )

    # Count unique relationship types per pair
    unique_counts = (
        df.groupby("sorted_pair")["Relationship"]
        .nunique()
        .reset_index(name="unique_relationship_types")
    )

    # Merge both summary counts
    df_final = df_final.merge(total_counts, on="sorted_pair", how="left")
    df_final = df_final.merge(unique_counts, on="sorted_pair", how="left")

    # Clean up and remove the helper column
    df_final = df_final.drop("sorted_pair", axis=1)
    df_final.drop_duplicates(inplace=True)

    print(f"Final cleaned DataFrame shape: {df_final.shape}")
    print("\nStandardized relationship types:")
    print(df_final["Relationship"].value_counts())

    print("\nSample of relationships with individual relationship type counts:")
    print(
        df_final[
            [
                "Entity1_ID",
                "Entity2_ID",
                "Relationship",
                "relationship_type_count",
                "total_relationship_count",
                "unique_relationship_types",
            ]
        ].head(10)
    )

    # Show examples of pairs with multiple instances of the same relationship type
    multiple_same_type = df_final[df_final["relationship_type_count"] > 1]
    if not multiple_same_type.empty:
        print(
            f"\nPairs with multiple instances of the same relationship type ({len(multiple_same_type)} rows):"
        )
        print(
            multiple_same_type[
                ["Entity1_ID", "Entity2_ID", "Relationship", "relationship_type_count"]
            ].head()
        )

    # Show relationship type distribution
    print(f"\nRelationship type count distribution:")
    print(df_final["relationship_type_count"].value_counts().sort_index())

    df_final.to_csv("relationships_with_counts.csv", index=False)

    # Create a separate summary table showing all relationship types for each pair
    relationship_pivot = df.pivot_table(
        index="sorted_pair",
        columns="Relationship",
        values="Entity1_ID",  # Just counting occurrences
        aggfunc="count",
        fill_value=0,
    ).reset_index()

    print(f"\nPivot table showing relationship type counts per pair (first 5 pairs):")
    print(relationship_pivot.head())

    relationship_pivot.to_csv("relationship_pivot_summary.csv", index=False)


# endregion
def draw_graph():
    # Load the cleaned relationships data
    try:
        df = pd.read_csv("./results/relationship_pivot_summary_by_chapter.csv")
    except FileNotFoundError:
        print(
            "Error: 'relationships_with_counts.csv' not found. Please run process_data() first."
        )
        return
    # Create a directed graph
    col_index = df.idxmax(axis=1, numeric_only=True)
    print(col_index)
    new_df = pd.DataFrame({"sorted_pair": df["sorted_pair"], "relationship": col_index})
    print(new_df)
    n_vertices = new_df["sorted_pair"].nunique()
    g = ig.Graph(directed=False)
    name_dict = load_names()
    print(name_dict)

    # Create a mapping from entity IDs to vertex indices
    entity_ids = list(name_dict.keys())
    g.add_vertices(len(entity_ids))
    g.vs["entity_id"] = entity_ids
    g.vs["name"] = [name_dict[entity_id] for entity_id in entity_ids]

    # Create a lookup for entity ID to vertex index
    id_to_index = {entity_id: i for i, entity_id in enumerate(entity_ids)}

    for _, row in new_df.iterrows():
        # Parse the tuple string properly
        sorted_pair_str = row["sorted_pair"]
        # Remove parentheses and quotes, then split
        cleaned_pair = sorted_pair_str.strip("()'").replace("'", "").replace('"', "")
        entity1, entity2 = [x.strip() for x in cleaned_pair.split(",")]
        relationship = row["relationship"]
        if relationship and entity1 in id_to_index and entity2 in id_to_index:
            g.add_edge(
                id_to_index[entity1],
                id_to_index[entity2],
                relationship=relationship.strip(),
            )
    fig, ax = plt.subplots(figsize=(12, 12))
    ig.plot(
        g,
        target=ax,
        vertex_size=80,
        vertex_color=["steelblue"],
        vertex_frame_width=4.0,
        vertex_frame_color="white",
        vertex_label=g.vs["name"],
        vertex_label_size=8.0,
        edge_label=[rel for rel in g.es["relationship"]],
        edge_label_size=6.0,
        edge_color="gray",
        edge_width=1.5,
        layout="fruchterman_reingold",
    )

    plt.show()

    # Save the graph to a GML file
    g.write_gml("character_relationships.gml")
    print("Graph saved as character_relationships.gml")


if __name__ == "__main__":
    draw_graph()
    process_data()
