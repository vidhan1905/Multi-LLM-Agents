import os
import sys
from dotenv import load_dotenv
from core.llm_manager import LLMManager
from agents.trip_planner import TripPlanner
from agents.pandas_agent import PandasAgent
from agents.sql_agent import SQLAgent
from agents.rag_agent import RAGAgent
from datetime import datetime

load_dotenv()

def save_to_markdown(plan_raw, filename=None):
    """Save the trip planning result to a Markdown file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trip_plan_{timestamp}.md"
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(plan_raw)
    
    return filename

class MultiLLMsApp:
    """
    Main application class for Multi-LLMs system.
    Provides a menu-driven interface for different AI agents.
    """
    
    def __init__(self):
        self.llm_manager = LLMManager()
        self.agents = {}
        self.initialize_agents()
        
    def initialize_agents(self):
        """Initialize available agents."""
        # For now, we'll keep the existing trip planner
        # Later we'll add more agents
        self.agents = {
            "trip_planner": {
                "name": "Trip Planning Agent",
                "description": "Plan complete travel itineraries with flights, accommodation, and activities",
                "class": TripPlanner,
                "available": True
            },
            "rag_agent": {
                "name": "RAG Agent",
                "description": "Retrieval-Augmented Generation for document Q&A",
                "class": RAGAgent,
                "available": True
            },
            "sql_agent": {
                "name": "Text-to-SQL Agent",
                "description": "Convert natural language to SQL queries",
                "class": SQLAgent,
                "available": True
            },
            "pandas_agent": {
                "name": "Pandas Agent",
                "description": "Data analysis and manipulation with pandas",
                "class": PandasAgent,
                "available": True
            }
        }
    
    def show_menu(self):
        """Display the main menu."""
        print("\n" + "="*60)
        print("🤖 Multi-LLMs System")
        print("="*60)
        print(f"Current LLM: {self.llm_manager.get_current_provider()} - {self.llm_manager.get_current_model()}")
        print(f"Temperature: {self.llm_manager.get_current_temperature()}")
        print("-"*60)
        
        for i, (key, agent) in enumerate(self.agents.items(), 1):
            status = "✅" if agent["available"] else "🚧"
            print(f"{i}. {status} {agent['name']}")
            print(f"   {agent['description']}")
            print()
        
        print(f"{len(self.agents) + 1}. ℹ️  System Status")
        print(f"{len(self.agents) + 2}. ❌ Exit")
        print("="*60)
        print("💡 To change LLM settings, edit llm_config.py file")
        print("="*60)
    
    def handle_choice(self, choice: str):
        """Handle user menu choice."""
        try:
            choice_num = int(choice)
            
            if choice_num == len(self.agents) + 1:
                self.show_system_status()
            elif choice_num == len(self.agents) + 2:
                print("👋 Goodbye!")
                sys.exit(0)
            elif 1 <= choice_num <= len(self.agents):
                agent_keys = list(self.agents.keys())
                agent_key = agent_keys[choice_num - 1]
                self.run_agent(agent_key)
            else:
                print("❌ Invalid choice. Please try again.")
                
        except ValueError:
            print("❌ Please enter a valid number.")
    
    def run_agent(self, agent_key: str):
        """Run a specific agent."""
        agent_info = self.agents[agent_key]
        
        if not agent_info["available"]:
            print(f"🚧 {agent_info['name']} is not yet implemented.")
            return
        
        print(f"\n🚀 Starting {agent_info['name']}...")
        
        if agent_key == "trip_planner":
            self.run_trip_planner()
        elif agent_key == "pandas_agent":
            self.run_pandas_agent()
        elif agent_key == "sql_agent":
            self.run_sql_agent()
        elif agent_key == "rag_agent":
            self.run_rag_agent()
        else:
            print(f"🚧 {agent_info['name']} implementation coming soon!")
    
    def run_pandas_agent(self):
        """Run the pandas agent."""
        try:
            pandas_agent = PandasAgent(llm_manager=self.llm_manager)
            if pandas_agent.agent_executor is None:
                return  # Warning is printed from the agent's constructor

            while True:
                print("\n" + "="*60)
                query = input("Enter your query for the pandas agent (or type 'exit' to return to menu): ")
                print("="*60)

                if query.lower() == 'exit':
                    break
                
                if not query:
                    print("Please enter a query.")
                    continue
                
                result = pandas_agent.run(query)
                
                print("\n" + "="*50)
                print("PANDAS AGENT RESULT")
                print("="*50)
                # The result from langchain agent executor is a dict with 'output' key
                print(result['output'] if isinstance(result, dict) and 'output' in result else result)
                print("="*50)

        except Exception as e:
            print(f"❌ An error occurred while running the pandas agent: {e}")
    
    def run_trip_planner(self):
        """Run the trip planning agent."""
        try:
            # Check for necessary environment variables
            required_env_vars = [
                "SERPER_API_KEY",
                "AMADEUS_CLIENT_ID",
                "AMADEUS_CLIENT_SECRET",
                "OPENWEATHERMAP_API_KEY"
            ]
            
            # Add LLM-specific API key
            required_env_vars.append(self.llm_manager.config["api_key_env"])
            
            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            if missing_vars:
                print(f"❌ Error: Missing environment variables: {', '.join(missing_vars)}")
                print("Please create a .env file in the root directory and add the required API keys.")
                return

            # Example travel request - customize as needed
            departure_str = "2025-08-01"
            return_str = "2025-08-06"
            
            departure_date = datetime.strptime(departure_str, "%Y-%m-%d")
            return_date = datetime.strptime(return_str, "%Y-%m-%d")
            duration = (return_date - departure_date).days
            
            travel_request = {
                "budget_in_inr": 100000.0,
                "travel_by": "flight",
                "travelers": {"adults": 2, "children": 1},
                "duration_days": duration,
                "departure_date": departure_str,
                "return_date": return_str,
                "departure_city": "Ahmedabad",
                "arrival_city": "Goa",
                "interests": ["adventure", "history", "food", "parks"],
                "special_requirements": ["need a stroller-friendly itinerary"],
                "travel_style": "mid-range",
                "accommodation_type": "hotel",
                "accommodation_budget_per_night": 20000.0,
                "nationality": "Indian"
            }

            print(f"🤖 Using LLM: {self.llm_manager.get_current_provider()} - {self.llm_manager.get_current_model()}")
            # Pass the LLM manager to TripPlanner
            trip_planner_crew = TripPlanner(llm_manager=self.llm_manager)
            result = trip_planner_crew.kickoff(inputs={"travel_request": travel_request})
            
            # Save to markdown file
            filename = save_to_markdown(result.raw if hasattr(result, 'raw') else str(result))
            print(f"✅ Trip plan saved to: {filename}")
            
            print("\n" + "="*50)
            print("TRIP PLANNING COMPLETE")
            print("="*50)
            print(f"📄 Full report saved to: {filename}")
            print(f"💰 Budget: ₹{travel_request['budget_in_inr']:,.2f}")
            print(f"✈️  Route: {travel_request['departure_city']} → {travel_request['arrival_city']}")
            print(f"📅 Duration: {travel_request['duration_days']} days")
            print(f"🤖 LLM: {self.llm_manager.get_current_provider()} - {self.llm_manager.get_current_model()}")
            print("="*50)

        except Exception as e:
            print(f"❌ An error occurred while running the trip planner: {e}")
    
    def run_sql_agent(self):
        """Run the SQL agent."""
        try:
            sql_agent = SQLAgent(llm_manager=self.llm_manager)
            if sql_agent.agent_executor is None:
                return  # Warning is printed from the agent's constructor

            while True:
                print("\n" + "="*60)
                query = input("Enter your query for the SQL agent (or type 'exit' to return to menu): ")
                print("="*60)

                if query.lower() == 'exit':
                    break
                
                if not query:
                    print("Please enter a query.")
                    continue
                
                result = sql_agent.run(query)
                
                print("\n" + "="*50)
                print("SQL AGENT RESULT")
                print("="*50)
                # The result from langchain agent executor is a dict with 'output' key
                print(result['output'] if isinstance(result, dict) and 'output' in result else result)
                print("="*50)

        except Exception as e:
            print(f"❌ An error occurred while running the SQL agent: {e}")
    
    def run_rag_agent(self):
        """Run the RAG agent."""
        try:
            rag_agent = RAGAgent(llm_manager=self.llm_manager)
            if rag_agent.retrieval_chain is None:
                return  # Warning is printed from the agent's constructor

            while True:
                print("\n" + "="*60)
                query = input("Enter your query for the RAG agent (or type 'exit' to return to menu): ")
                print("="*60)

                if query.lower() == 'exit':
                    break
                
                if not query:
                    print("Please enter a query.")
                    continue
                
                result = rag_agent.run(query)
                
                print("\n" + "="*50)
                print("RAG AGENT RESULT")
                print("="*50)
                print(result)
                print("="*50)

        except Exception as e:
            print(f"❌ An error occurred while running the RAG agent: {e}")
    
    def show_system_status(self):
        """Show system status and configuration."""
        print("\n📊 System Status:")
        print("-" * 30)
        print(f"LLM Provider: {self.llm_manager.get_current_provider()}")
        print(f"Model: {self.llm_manager.get_current_model()}")
        print(f"Temperature: {self.llm_manager.get_current_temperature()}")
        
        print("\n🔧 Available Agents:")
        for key, agent in self.agents.items():
            status = "✅ Ready" if agent["available"] else "🚧 Not Implemented"
            framework = "CrewAI" if key == "trip_planner" else "LangChain"
            print(f"  - {agent['name']}: {status} ({framework})")
        
        print("\n📁 Configuration Files:")
        config_files = ["llm_config.py", "config/agents.yaml", "config/tasks.yaml"]
        for file in config_files:
            exists = "✅" if os.path.exists(file) else "❌"
            print(f"  {exists} {file}")
        
        print("\n💡 To change LLM settings:")
        print("  1. Edit llm_config.py file")
        print("  2. Change CURRENT_PROVIDER, model, temperature, etc.")
        print("  3. Restart the application")
        
        print("\n🎯 Framework Usage:")
        print("  - Trip Planning Agent: CrewAI (multi-agent orchestration)")
        print("  - RAG Agent: LangChain (document processing)")
        print("  - SQL Agent: LangChain (database queries)")
        print("  - Pandas Agent: LangChain (data analysis)")
    
    def run(self):
        """Main application loop."""
        print("🚀 Starting Multi-LLMs System...")
        
        while True:
            try:
                self.show_menu()
                choice = input("Select an option: ")
                self.handle_choice(choice)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                sys.exit(0)
            except Exception as e:
                print(f"❌ An error occurred: {e}")

def main():
    """Main entry point."""
    app = MultiLLMsApp()
    app.run()

if __name__ == "__main__":
    main() 