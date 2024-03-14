from neo4j import GraphDatabase

def test_neo4j_connection(uri, username, password):
    try:
        # Connect to the Neo4j database
        with GraphDatabase.driver(uri, auth=(username, password)) as driver:
            # Use a session to execute a simple query
            with driver.session() as session:
                result = session.run("RETURN 'Connected to Neo4j' AS message")
                record = result.single()
                message = record["message"]
                print(message)
                return True

    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return False

# Replace these with your actual Neo4j connection details
neo4j_uri = "bolt://localhost:7687"
# http://localhost:7474/browser/
neo4j_username = "neo4j"
neo4j_password = "password"
# neo4j_uri = "neo4j://vm0.node-i2olpfesyb66o.eastus2.cloudapp.azure.com:7687"
# neo4j_username = "neo4j"
# neo4j_password = "SuketaNeo4jp@ss1"

# Test the Neo4j connection
connection_successful = test_neo4j_connection(neo4j_uri, neo4j_username, neo4j_password)

if connection_successful:
    print("Neo4j connection test successful!")
else:
    print("Neo4j connection test failed.")
