import configparser
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)
DATA_DIR = BASE_DIR / "data" / "European_LV_CSV"

# ==============================================================================
# 1. LOAD & CLEAN DATA
# ==============================================================================
df_buscoords = pd.read_csv(DATA_DIR / "Buscoords.csv")
df_lineCodes = pd.read_csv(DATA_DIR / "LineCodes.csv")
df_lines = pd.read_csv(DATA_DIR / "Lines.csv")
df_loads = pd.read_csv(DATA_DIR / "Loads.csv")
df_loadShapes = pd.read_csv(DATA_DIR / "LoadShapes.csv")
df_transformer = pd.read_csv(DATA_DIR / "Transformer.csv")

for df in [df_buscoords, df_lineCodes, df_lines, df_loads, df_loadShapes, df_transformer]:
    df.columns = df.columns.str.strip()

# Source.csv is NOT a tabular CSV -- it's an INI-style block ([Source] / key=value),
# so it needs configparser, not pd.read_csv (which was silently wrong before).
def parse_source_ini(path: Path) -> dict:
    cp = configparser.ConfigParser()
    cp.read(path)
    section = cp[cp.sections()[0]]

    def num(key, default=None):
        raw = section.get(key, fallback=None)
        if raw is None:
            return default
        return float(raw.strip().split()[0])  # strips units like "11 kV" / "3000 A"

    return {
        "voltage_kv": num("Voltage"),
        "pu": num("pu", 1.0),
        "isc3": num("ISC3"),
        "isc1": num("ISC1"),
    }

source_data = parse_source_ini(DATA_DIR / "Source.csv")

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
        if cls.driver is None:
            print(f"Initializing global Neo4j driver connection to {uri}...")
            try:
                cls.driver = GraphDatabase.driver(uri, auth=(user, password))
                cls.driver.verify_connectivity()
                print("Global Neo4j driver successfully initialized.")
            except Exception as e:
                print(f"Failed to connect to Neo4j. Error: {e}")
                cls.driver = None
                raise
        return cls.driver

    @classmethod
    def get_driver(cls):
        if cls.driver is None:
            raise RuntimeError("Neo4jConnector driver has not been initialized. Call initialize() first.")
        return cls.driver

    @classmethod
    def close(cls):
        if cls.driver is not None:
            print("Closing global Neo4j driver pool...")
            cls.driver.close()
            cls.driver = None
            print("Global Neo4j driver closed successfully.")


def run_query(query, parameters=None):
    driver = Neo4jConnector.get_driver()
    with driver.session() as session:
        session.run(query, parameters)


def setup_constraints():
    print("Verifying database schema constraints...")
    run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (b:Bus) REQUIRE b.id IS UNIQUE")
    run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (l:Load) REQUIRE l.id IS UNIQUE")
    run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (lc:LineCode) REQUIRE lc.name IS UNIQUE")
    run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (t:SubstationTransformer) REQUIRE t.id IS UNIQUE")
    run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Source) REQUIRE g.id IS UNIQUE")
    print("Database constraints are fully set up.")


# ==============================================================================
# 3. KNOWLEDGE GRAPH ETL INGESTION FUNCTIONS
# ==============================================================================

def ingest_knowledge_graph():
    print("Ingesting Knowledge Graph...")

    # A. Ingest :Bus Nodes
    print("- Creating Bus Nodes...")
    run_query("""
    UNWIND $batch AS row
    MERGE (b:Bus {id: toString(row.Busname)})
    SET b.x = toFloat(row.x),
        b.y = toFloat(row.y)
    """, {"batch": buscoords_list})

    # B. Ingest :LineCode Nodes
    print("- Creating LineCode Nodes...")
    run_query("""
    UNWIND $batch AS row
    MERGE (lc:LineCode {name: toString(row.Name)})
    SET lc.nphases = toInteger(row.nphases),
        lc.r1 = toFloat(row.R1),
        lc.x1 = toFloat(row.X1),
        lc.r0 = toFloat(row.R0),
        lc.x0 = toFloat(row.X0),
        lc.c1 = toFloat(row.C1),
        lc.c0 = toFloat(row.C0),
        lc.units = toString(row.Units)
    """, {"batch": linecodes_list})

    # C. Ingest :Source Node (from Source.csv)
    print("- Creating Source (Source) Node...")
    run_query("""
    MERGE (g:Source {id: 'SourceBus'})
    SET g.nominalKV = toFloat($voltage_kv),
        g.pu = toFloat($pu),
        g.isc3 = toFloat($isc3),
        g.isc1 = toFloat($isc1)
    """, source_data)

    # D. Ingest :Load Nodes & [:SUPPLIES_POWER] Edges
    print("- Creating Load Nodes and Relationships...")
    run_query("""
    UNWIND $batch AS row
    MERGE (l:Load {id: toString(row.Name)})
    SET l.numPhases = toInteger(row.numPhases),
        l.targetPhase = toString(row.phases),
        l.baseKW = toFloat(row.kW),
        l.kV = toFloat(row.kV),
        l.model = toInteger(row.Model),
        l.powerFactor = toFloat(row.PF),
        l.connectionType = toString(row.Connection),
        l.profileURI = toString(row.Yearly)

    WITH l, row
    MATCH (b:Bus {id: toString(row.Bus)})
    MERGE (b)-[:SUPPLIES_POWER]->(l)
    """, {"batch": loads_list})

    # E. Ingest :SubstationTransformer Nodes & Edges
    print("- Creating SubstationTransformer Nodes and Relationships...")
    run_query("""
    UNWIND $batch AS row
    MERGE (t:SubstationTransformer {id: toString(row.Name)})
    SET t.phases = toInteger(row.phases),
        t.ratedMVA = toFloat(row.MVA),
        t.primaryKV = toFloat(row.kV_pri),
        t.secondaryKV = toFloat(row.kV_sec),
        t.primaryConn = toString(row.Conn_pri),
        t.secondaryConn = toString(row.Conn_sec),
        t.xhl = toFloat(row.`%XHL`),
        t.resistancePct = toFloat(row.`% resistance`)

    WITH t, row
    MATCH (g:Source {id: toString(row.bus1)})
    MERGE (g)-[:FEEDS]->(t)

    WITH t, row
    MATCH (b:Bus {id: toString(row.bus2)})
    MERGE (t)-[:HAS_FEEDER_HEAD]->(b)
    """, {"batch": transformer_list})

    # F. Ingest [:LINE_SEGMENT] Topology and [:CONFORMS_TO] Edge Context
    print("- Linking Bus Topology and Line Specifications...")
    run_query("""
    UNWIND $batch AS row
    MATCH (b1:Bus {id: toString(row.Bus1)})
    MATCH (b2:Bus {id: toString(row.Bus2)})
    MATCH (lc:LineCode {name: toString(row.LineCode)})

    MERGE (b1)-[r:LINE_SEGMENT {name: toString(row.Name)}]->(b2)
    SET r.length = toFloat(row.Length),
        r.units = toString(row.Units),
        r.phases = toString(row.Phases)

    MERGE (b2)-[:CONFORMS_TO]->(lc)
    """, {"batch": lines_list})

    print("Knowledge Graph Ingestion Complete!")


def add_derived_relationships():
    """Post-processing step (from the project handoff, previously never run):
    precomputed shortcut relationships that give the GNN long-range signal
    a 2-3 layer message-passing model wouldn't otherwise reach in one pass."""

    print("Adding derived relationships for GNN feature engineering...")

    print("- [:UPSTREAM_OF] full downstream hierarchy from the feeder head...")
    run_query("""
        MATCH (t:SubstationTransformer)-[:HAS_FEEDER_HEAD]->(root:Bus)
        MATCH path = (root)-[:LINE_SEGMENT*]->(target:Bus)
        WITH nodes(path) AS busNodes
        UNWIND range(0, size(busNodes)-2) AS i
        UNWIND range(i+1, size(busNodes)-1) AS j
        WITH busNodes[i] AS upstreamBus, busNodes[j] AS downstreamBus
        MERGE (upstreamBus)-[:UPSTREAM_OF]->(downstreamBus)
        """)

    print("- Bus.hopsFromSource and Bus.cumulativeDownstreamKW (GNN node features)...")
    run_query("""
        MATCH (t:SubstationTransformer)-[:HAS_FEEDER_HEAD]->(root:Bus)
        MATCH path = (root)-[:LINE_SEGMENT*0..]->(b:Bus)
        WITH b, min(length(path)) AS hops
        SET b.hopsFromSource = hops
        """)
    run_query("""
        MATCH (b:Bus)
        OPTIONAL MATCH (b)-[:UPSTREAM_OF]->(:Bus)-[:SUPPLIES_POWER]-(l:Load)
        OPTIONAL MATCH (b)-[:SUPPLIES_POWER]-(ownLoad:Load)
        WITH b, coalesce(sum(l.baseKW), 0) + coalesce(sum(ownLoad.baseKW), 0) AS cumKW
        SET b.cumulativeDownstreamKW = cumKW
        """)

    print("Derived relationships complete!")


# ==============================================================================
# 4. RUN PIPELINE
# ==============================================================================
if __name__ == "__main__":
    URI = os.environ.get("NEO4J_URI")
    USER = os.environ.get("NEO4J_USERNAME")
    PASSWORD = os.environ.get("NEO4J_PASSWORD")

    print("Connecting to Neo4j...")
    Neo4jConnector.initialize(uri=URI, user=USER, password=PASSWORD)

    setup_constraints()
    ingest_knowledge_graph()
    add_derived_relationships()

    Neo4jConnector.close()