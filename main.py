import os
import csv
from neo4jdata import driver, execute_cypher, close_connection

folder_path = "twitter"

# Function to process a specific node ID and load its data
def load_and_insert_data(node_id):
    # Load data from files
    edges = load_edges(node_id)
    circles = load_circles(node_id)
    features = load_node_features(node_id)
    ego_features = load_ego_features(node_id)
    featnames = load_feature_names(node_id)

    # Create nodes and relationships
    create_user_node(node_id, features, ego_features)
    create_edges(node_id, edges)
    create_circles(node_id, circles)
    create_features(node_id, features)


# Function to load nodes (e.g., featnames) for a specific node_id
def load_feature_names(node_id):
    featnames_file = os.path.join(folder_path, f"{node_id}.featnames")
    featnames = []

    with open(featnames_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            featnames.append(row[0])  # Assuming it's one feature name per line

    return featnames


# Function to load features for a specific node_id
def load_node_features(node_id):
    feat_file = os.path.join(folder_path, f"{node_id}.feat")
    features = []

    with open(feat_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            features.append(row)  # Mỗi row có thể là một danh sách con

    flattened_features = [item for sublist in features for item in sublist]

    return flattened_features



# Function to load ego features for a specific node_id
def load_ego_features(node_id):
    egofeat_file = os.path.join(folder_path, f"{node_id}.egofeat")
    ego_features = []

    with open(egofeat_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            ego_features.append(row)


    flattened_ego_features = [item for sublist in ego_features for item in sublist]

    return flattened_ego_features


# Function to load edges (relationships) for a specific node_id
def load_edges(node_id):
    edges_file = os.path.join(folder_path, f"{node_id}.edges")
    edges = []

    with open(edges_file, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            edges.append((node_id, row[0]))  # node_id -> other node (for Facebook, you can add bidirectional)

    return edges


# Function to load circles (grouping) for a specific node_id
def load_circles(node_id):
    circles_file = os.path.join(folder_path, f"{node_id}.circles")
    circles = []

    with open(circles_file, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            circle_name = row[0]
            nodes_in_circle = row[1:]  # nodes in this circle
            circles.append((circle_name, nodes_in_circle))

    return circles


def create_user_node(node_id, features, ego_features):
    query = """
    MERGE (u:User {id: $node_id})
    SET u.features = $features, u.ego_features = $ego_features
    """
    params = {
        'node_id': node_id,
        'features': features,
        'ego_features': ego_features
    }
    execute_cypher(query, params)


def create_edges(node_id, edges):
    for target_id in edges:
        query = """
        MATCH (u:User {id: $node_id}), (v:User {id: $target_id})
        MERGE (u)-[:FOLLOWS]->(v)
        """
        params = {
            'node_id': node_id,
            'target_id': target_id
        }
        execute_cypher(query, params)


def create_circles(node_id, circles):
    for circle_name, nodes_in_circle in circles:
        for target_id in nodes_in_circle:
            query = """
            MATCH (u:User {id: $node_id}), (v:User {id: $target_id})
            MERGE (u)-[:IN_CIRCLE]->(v)
            SET u.circle_name = $circle_name
            """
            params = {
                'node_id': node_id,
                'target_id': target_id,
                'circle_name': circle_name
            }
            execute_cypher(query, params)

def create_features(node_id, features):
    query = """
    MATCH (u:User {id: $node_id})
    SET u.features = $features
    """
    params = {
        'node_id': node_id,
        'features': features
    }
    execute_cypher(query, params)



# Function to process all nodes in a folder
def load_all_data_from_folder(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Find unique node IDs based on files
    node_ids = set()
    for file in files:
        if file.endswith(('.featnames', '.edges', '.circles', '.feat', '.egofeat')):
            node_id = file.split('.')[0]  # Extract the node ID from the filename
            node_ids.add(node_id)

    # Load data for each unique node ID
    for node_id in node_ids:
        load_and_insert_data(node_id)


# Main function to load all data
def main():
    load_all_data_from_folder(folder_path)


# Execute the main function
if __name__ == "__main__":
    main()
    close_connection()  # Ensure to close the connection after operations
