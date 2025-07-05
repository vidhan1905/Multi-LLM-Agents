import os
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from core.llm_manager import LLMManager

class PandasAgent:
    """Agent that can interact with one or more Pandas DataFrames."""
    
    def __init__(self, llm_manager: LLMManager, csv_dir: str = "data"):
        """
        Initializes the PandasAgent.
        
        Args:
            llm_manager: An instance of LLMManager to get the configured LLM.
            csv_dir: The directory containing the CSV files to load.
        """
        self.llm = llm_manager.get_langchain_llm()
        self.df_list, self.df_names = self._load_dataframes(csv_dir)
        
        if not self.df_list:
            print(f"Warning: No CSV files found in directory: {csv_dir}. The agent will not have any data to query.")
            self.agent_executor = None
        else:
            # If there's only one dataframe, pass it directly. Otherwise, pass the list.
            dfs_to_pass = self.df_list[0] if len(self.df_list) == 1 else self.df_list
            
            # Custom prompt prefix to guide the LLM
            prefix_lines = [
                "You are working with one or more pandas dataframes in Python.",
                "The dataframes are already loaded in memory and are available for you to use.",
                "You should not try to redefine them.",
                "When you output the action, you must not use brackets. For example, the action should be `python_repl_ast` and not `[python_repl_ast]`."
            ]
            if len(self.df_list) > 1:
                prefix_lines.append("\\nThe following dataframes are available:")
                for i, name in enumerate(self.df_names):
                    prefix_lines.append(f"- df{i+1}: A dataframe loaded from '{name}.csv'")
            prefix = "\\n".join(prefix_lines)

            self.agent_executor = create_pandas_dataframe_agent(
                self.llm,
                dfs_to_pass,
                prefix=prefix,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                allow_dangerous_code=True,
                handle_parsing_errors=True
            )

    def _load_dataframes(self, csv_dir: str):
        """Loads all CSV files from a directory into a list of pandas DataFrames and their names."""
        df_list = []
        df_names = []
        if not os.path.exists(csv_dir):
            print(f"Directory not found: {csv_dir}")
            return df_list, df_names
            
        filenames = sorted([f for f in os.listdir(csv_dir) if f.endswith(".csv")])
        for filename in filenames:
            file_path = os.path.join(csv_dir, filename)
            df_name = os.path.splitext(filename)[0]
            print(f"Loading dataframe '{df_name}' from: {file_path}")
            try:
                df_list.append(pd.read_csv(file_path))
                df_names.append(df_name)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        return df_list, df_names

    def run(self, query: str):
        """
        Runs a query against the loaded dataframes.
        
        Args:
            query: The natural language query to execute.
            
        Returns:
            The result of the agent's execution.
        """
        if self.agent_executor is None:
            return "No dataframes loaded. Please add CSV files to the 'data' directory."
        
        try:
            return self.agent_executor.invoke(query)
        except Exception as e:
            return f"An error occurred: {e}"
