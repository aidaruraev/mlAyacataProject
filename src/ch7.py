import pandas as pd
import seaborn as sns
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics.pairwise import cosine_similarity


# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)


# Create a customer-item matrix
def create_customer_item_matrix(df):
    customer_item_matrix = df.pivot_table(
        index='CustomerID',
        columns='StockCode',
        values='Quantity',
        aggfunc='sum'
    )
    # Convert quantities to binary (1 if purchased, 0 otherwise)
    return customer_item_matrix.map(lambda x: 1 if x > 0 else 0)


# Generate frequent itemsets using the Apriori algorithm
def generate_frequent_itemsets(customer_item_matrix, min_support=0.03):
    frequent_items = apriori(customer_item_matrix, min_support=min_support, use_colnames=True)
    frequent_items["n_items"] = frequent_items["itemsets"].apply(lambda x: len(x))
    return frequent_items


# Generate association rules from the frequent itemsets
def generate_association_rules(frequent_items, metric="confidence", min_threshold=0.6):
    return association_rules(frequent_items, metric=metric, min_threshold=min_threshold)


# Get the top 20 items by lift and confidence
def get_top_items_by_lift_and_confidence(rules):
    most_lift = rules.sort_values(by="lift", ascending=False).head(20).pivot_table(
        index='antecedents',
        columns='consequents',
        values='lift',
        aggfunc='sum'
    )
    most_conf = rules.sort_values(by="confidence", ascending=False).head(20).pivot_table(
        index='antecedents',
        columns='consequents',
        values='confidence',
        aggfunc='sum'
    )
    return most_lift, most_conf


# Create a user-user similarity matrix
def create_user_user_similarity_matrix(customer_item_matrix):
    user_user_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix))
    user_user_sim_matrix.columns = customer_item_matrix.index
    user_user_sim_matrix['CustomerID'] = customer_item_matrix.index
    return user_user_sim_matrix.set_index('CustomerID')


# Get potential recommendation items based on a target customer
def get_recommendations_by_similarity(df, user_user_sim_matrix, target_customer_id):
    top10_similar_users = user_user_sim_matrix.loc[target_customer_id].sort_values(ascending=False).head(11).to_dict()
    potential_rec_items = {}

    for user, cos_sim in top10_similar_users.items():
        if user == target_customer_id:
            continue

        items_bought_by_sim = set(df.loc[df["CustomerID"] == user]["StockCode"].unique())

        for each_item in items_bought_by_sim:
            if each_item not in potential_rec_items:
                potential_rec_items[each_item] = 0
            potential_rec_items[each_item] += cos_sim

    # Sort recommendations by similarity score
    potential_rec_items = sorted(potential_rec_items.items(), key=lambda x: x[1], reverse=True)
    return potential_rec_items


# Get most similar items based on item-item similarity
def get_most_similar_items(customer_item_matrix, stock_code):
    item_item_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix.T),
                                        index=customer_item_matrix.columns,
                                        columns=customer_item_matrix.columns)
    return item_item_sim_matrix.loc[stock_code].sort_values(ascending=False).head(10)


# Main execution flow
def main():
    # Load data
    df = load_data("../data/data.csv")

    # Create customer-item matrix
    customer_item_matrix = create_customer_item_matrix(df)

    # Generate frequent itemsets and association rules
    frequent_items = generate_frequent_itemsets(customer_item_matrix)
    rules = generate_association_rules(frequent_items)

    # Get top items by lift and confidence
    most_lift, most_conf = get_top_items_by_lift_and_confidence(rules)

    # Create user-user similarity matrix
    user_user_sim_matrix = create_user_user_similarity_matrix(customer_item_matrix)

    # Define target customer
    TARGET_CUSTOMER = 14806.0

    # Get potential recommendations based on similar users
    potential_rec_items = get_recommendations_by_similarity(df, user_user_sim_matrix, TARGET_CUSTOMER)

    # Print potential recommendations
    print("Potential recommendations based on user similarity:")
    for item, score in potential_rec_items[:10]:
        print(f"Item: {item}, Score: {score}")

    # Get and print most similar items for a specific stock code
    stock_code = "23166"
    most_similar_items = get_most_similar_items(customer_item_matrix, stock_code)

    print(f"\nMost similar items to {stock_code}:")
    print(most_similar_items)


if __name__ == "__main__":
    main()
