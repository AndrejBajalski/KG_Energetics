import os
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
from neo4j import GraphDatabase

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)
# ==============================================================================
# 1. LOAD & CLEAN DATA
# ==============================================================================
# Read input CSV files
df_buscoords = pd.read_csv("../data/European_LV_CSV/Buscoords.csv")
df_lineCodes = pd.read_csv("../data/European_LV_CSV/LineCodes.csv")
df_lines = pd.read_csv("../data/European_LV_CSV/Lines.csv")
df_loads = pd.read_csv("../data/European_LV_CSV/Loads.csv")
df_loadShapes = pd.read_csv("../data/European_LV_CSV/LoadShapes.csv")
df_source = pd.read_csv("../data/European_LV_CSV/Source.csv")
df_transformer = pd.read_csv("../data/European_LV_CSV/Transformer.csv")

# Clean up column spaces if any exist in the CSV headers
for df in [df_buscoords, df_lineCodes, df_lines, df_loads, df_loadShapes, df_source, df_transformer]:
    df.columns = df.columns.str.strip()

# Convert all dataframes to dictionaries for optimized Neo4j batch insertion
buscoords_list = df_buscoords.to_dict(orient="records")
linecodes_list = df_lineCodes.to_dict(orient="records")
lines_list = df_lines.to_dict(orient="records")
loads_list = df_loads.to_dict(orient="records")
transformer_list = df_transformer.to_dict(orient="records")

# ==============================================================================
# 2. NEO4J CONNECTION SETUP
# ==============================================================================
class Neo4jConnector:
    """
    Thread-safe Singleton class to manage a single shared connection pool
    to the Neo4j Graph Database across the entire application workspace.
    """
    _instance = None
    driver = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Neo4jConnector, cls).__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, uri, user, password):
        """Initializes the underlying Neo4j driver instance once."""
        if cls.driver is None:
            print(f"Initializing global Neo4j driver connection to {uri}...")
            try:
                cls.driver = GraphDatabase.driver(uri, auth=(user, password))
                # Explicitly test the connection pool on initialization
                cls.driver.verify_connectivity()
                print("Global Neo4j driver successfully initialized.")
            except Exception as e:
                print(f"Failed to connect to Neo4j. Error: {e}")
                cls.driver = None
                raise
        return cls.driver

    @classmethod
    def get_driver(cls):
        """Retrieves the active driver singleton instance."""
        if cls.driver is None:
            raise RuntimeError("Neo4jConnector driver has not been initialized. Call initialize() first.")
        return cls.driver

    @classmethod
    def close(cls):
        """Closes the active connection pool cleanly."""
        if cls.driver is not None:
            print("Closing global Neo4j driver pool...")
            cls.driver.close()
            cls.driver = None
            print("Global Neo4j driver closed successfully.")


# Shortcut helper function to query using the singleton context
def run_query(query, parameters=None):
    driver = Neo4jConnector.get_driver()
    with driver.session() as session:
        session.run(query, parameters)

# Create Constraints to optimize merge operations and guarantee uniqueness
def setup_constraints():
    print("Verifying database schema constraints...")
    run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (b:Bus) REQUIRE b.id IS UNIQUE")
    run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (l:Load) REQUIRE l.id IS UNIQUE")
    run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (lc:LineCode) REQUIRE lc.name IS UNIQUE")
    run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (t:SubstationTransformer) REQUIRE t.id IS UNIQUE")
    print("Database constraints are fully set up.")


# ==============================================================================
# 3. KNOWLEDGE GRAPH ETL INGESTION FUNCTIONS
# ==============================================================================

def ingest_knowledge_graph():
    print("Ingesting Knowledge Graph...")

    # A. Ingest :Bus Nodes
    print("- Creating Bus Nodes...")
    bus_query = """
    UNWIND $batch AS row
    MERGE (b:Bus {id: toString(row.Busname)})
    SET b.x = toFloat(row.x),
        b.y = toFloat(row.y)
    """
    run_query(bus_query, {"batch": buscoords_list})

    # B. Ingest :LineCode Nodes
    print("- Creating LineCode Nodes...")
    linecode_query = """
    UNWIND $batch AS row
    MERGE (lc:LineCode {name: toString(row.Name)})
    SET lc.nphases = toInteger(row.nphases),
        lc.r1 = toFloat(row.r1),
        lc.x1 = toFloat(row.x1),
        lc.r0 = toFloat(row.r0),
        lc.x0 = toFloat(row.x0),
        lc.c1 = toFloat(row.c1),
        lc.c0 = toFloat(row.c0),
        lc.units = toString(row.units)
    """
    run_query(linecode_query, {"batch": linecodes_list})

    # C. Ingest :EnergyConsumer (Load) Nodes & [:SUPPLIES_POWER] Edges
    print("- Creating EnergyConsumer Nodes and Relationships...")
    load_query = """
    UNWIND $batch AS row
    // Create the Consumer Node
    MERGE (l:Load {id: toString(row.Name)})
    SET l.numPhases = toInteger(row.numPhases),
        l.targetPhase = toString(row.phases),
        l.baseKW = toFloat(row.kW),
        l.powerFactor = toFloat(row.PF),
        l.connectionType = toString(row.Connection),
        l.profileURI = toString(row.Yearly)

    // Connect to the hosting topological Bus
    WITH l, row
    MATCH (b:Bus {id: toString(row.Bus)})
    MERGE (b)-[:SUPPLIES_POWER]->(l)
    """
    run_query(load_query, {"batch": loads_list})

    # D. Ingest :SubstationTransformer Nodes & [:HAS_FEEDER_HEAD] Edges
    print("- Creating SubstationTransformer Nodes and Relationships...")
    transformer_query = """
    UNWIND $batch AS row
    MERGE (t:SubstationTransformer {id: toString(row.Name)})
    SET t.ratedMVA = toFloat(row.MVA),
        t.primaryKV = toFloat(row.kV_primary),
        t.secondaryKV = toFloat(row.kV_secondary),
        t.connectionType = toString(row.Connection),
        t.resistance = toFloat(row.R_pct),
        t.reactance = toFloat(row.X_pct)

    // Connect transformer to its secondary low-voltage secondary bus head
    WITH t, row
    MATCH (b:Bus {id: toString(row.Bus_secondary)})
    MERGE (t)-[:HAS_FEEDER_HEAD]->(b)
    """
    run_query(transformer_query, {"batch": transformer_list})

    # E. Ingest [:LINE_SEGMENT] Topology and Edge Context [:CONFORMS_TO]
    print("- Linking Bus Topology and Line Specifications...")
    topology_query = """
    UNWIND $batch AS row
    MATCH (b1:Bus {id: toString(row.Bus1)})
    MATCH (b2:Bus {id: toString(row.Bus2)})
    MATCH (lc:LineCode {name: toString(row.LineCode)})

    // Create Directed Physical Line Relationship
    MERGE (b1)-[r:LINE_SEGMENT {name: toString(row.Name)}]->(b2)
    SET r.length = toFloat(row.Length),
        r.units = toString(row.Units),
        r.phases = toString(row.Phases)

    // Link relationship context to LineCode specs
    MERGE (b2)-[:CONFORMS_TO]->(lc)
    """
    run_query(topology_query, {"batch": lines_list})

    print("Knowledge Graph Ingestion Complete!")


# ==============================================================================
# 4. RUN PIPELINE
# ==============================================================================
if __name__ == "__main__":

    URI = os.environ.get("NEO4J_URI")
    USER = os.environ.get("NEO4J_USERNAME")
    PASSWORD = os.environ.get("NEO4J_PASSWORD")
    # Initialize the global connector singleton block once
    print("Connecting to Neo4j...")
    Neo4jConnector.initialize(
        uri=URI,
        user=USER,
        password=PASSWORD
    )
    # Execute schema configuration
    setup_constraints()
    # Stream dictionaries to Neo4j database instance via singleton instance
    ingest_knowledge_graph()
    # Close the active connection pool cleanly
    Neo4jConnector.close()