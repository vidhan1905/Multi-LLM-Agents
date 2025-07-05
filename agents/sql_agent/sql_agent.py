import os
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from core.llm_manager import LLMManager

class SQLAgent:
    """Agent that can interact with a SQL Database."""

    def __init__(self, llm_manager: LLMManager):
        """
        Initializes the SQLAgent.
        
        Args:
            llm_manager: An instance of LLMManager to get the configured LLM.
        """
        self.llm = llm_manager.get_langchain_llm()
        self.db = self._connect_to_db()
        
        if self.db:
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            self.agent_executor = create_sql_agent(
                llm=self.llm,
                toolkit=toolkit,
                verbose=True,
                handle_parsing_errors=True
            )
        else:
            self.agent_executor = None

    def _connect_to_db(self):
        """Connects to the PostgreSQL database using environment variables."""
        try:
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT")
            db_name = os.getenv("DB_NAME")

            if not all([db_user, db_password, db_host, db_port, db_name]):
                print("Warning: Missing one or more database environment variables (DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME). SQL Agent will be disabled.")
                return None

            pg_uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            print("Connecting to PostgreSQL database...")
            db = SQLDatabase.from_uri(pg_uri)
            print("Database connection successful.")
            return db
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            return None

    def run(self, query: str):
        """
        Runs a query against the database.
        
        Args:
            query: The natural language query to execute.
            
        Returns:
            The result of the agent's execution.
        """
        if self.agent_executor is None:
            return "SQL Agent is not available due to a missing database connection. Please check your .env file."
        
        try:
            return self.agent_executor.invoke({"input": query})
        except Exception as e:
            return f"An error occurred: {e}"
