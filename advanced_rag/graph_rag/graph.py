import logging
import os

from neo4j import GraphDatabase

from config import NEO4J_URI, NEO4J_DATABASE

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    def __init__(self):
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "Password")

        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(username, password))
        self.database = NEO4J_DATABASE

        logger.info("Neo4j driver initialized")


    def close(self):
        self.driver.close()
        logger.info("Neo4j connection closed")


    def verify_connection(self):
        self.driver.verify_connectivity()
        logger.info("Successfully connected to Neo4j")


    def clear_graph(self):
        query = "MATCH (n) DETACH DELETE n"

        with self.driver.session(database=self.database) as session:
            session.run(query)

        logger.info("Knowledge graph cleared")


    def add_entity(self, name, entity_type):
        query = """
        MERGE (n:Entity {name: $name})
        SET n.type = $entity_type
        """

        with self.driver.session(database=self.database) as session:
            session.run(query, name=name, entity_type=entity_type)

        logger.info("Added entity: %s (%s)", name, entity_type)


    def add_relationship(self, source, relationship, target):
        query = """
        MERGE (source:Entity {name: $source})
        MERGE (target:Entity {name: $target})
        MERGE (source)-[r:RELATES_TO {type: $relationship}]->(target)
        """

        with self.driver.session(database=self.database) as session:
            session.run(query, source=source, relationship=relationship, target=target)

        logger.info("Added relationship: %s -[%s]-> %s", source, relationship, target)


    def search_entity(self, entity_name):
        query = """
        MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
        WHERE toLower(source.name) CONTAINS toLower($entity_name)
           OR toLower(target.name) CONTAINS toLower($entity_name)
        RETURN source.name AS source, r.type AS relationship, target.name AS target
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, entity_name=entity_name)
            records = list(result)

        logger.info("Found %d graph relationships for entity: %s", len(records), entity_name)

        return [(record["source"], record["relationship"], record["target"]) for record in records]