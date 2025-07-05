from __future__ import annotations
import os, requests, yaml
from typing import Any, Dict, Type, Optional, List
from pydantic import BaseModel, Field, validator, field_validator, ValidationInfo
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from crewai import LLM
from crewai_tools import SerperDevTool
from crewai.tools import BaseTool
from datetime import datetime
import time
from functools import wraps

from langchain_community.utilities import OpenWeatherMapAPIWrapper, GoogleSerperAPIWrapper
from core.llm_manager import LLMManager


class AmadeusInput(BaseModel):
    origin: str = Field(..., description="Origin IATA code")
    destination: str = Field(..., description="Destination IATA code")
    departure_date: str = Field(..., description="Departure date in YYYY-MM-DD format")
    return_date: str | None = Field(None, description="Return date in YYYY-MM-DD format (optional)")
    adults: int = Field(1, description="Number of adult passengers")

class AmadeusFlightsTool(BaseTool):
    name: str = "amadeus_flights_search"
    description: str = "Search for flight offers using Amadeus API."
    args_schema: Type[BaseModel] = AmadeusInput

    def get_token(self) -> str:
        url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "client_credentials",
            "client_id": os.getenv("AMADEUS_CLIENT_ID"),
            "client_secret": os.getenv("AMADEUS_CLIENT_SECRET")
        }
        r = requests.post(url, headers=headers, data=data)
        if r.status_code != 200:
            raise Exception(f"Auth Error: {r.text}")
        return r.json()["access_token"]

    def _run(self, origin: str, destination: str, departure_date: str,
         return_date: str | None = None, adults: int = 1) -> Dict[str, Any]:
        token = self.get_token()
        url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
        headers = {"Authorization": f"Bearer {token}"}
        params = {
            "originLocationCode": origin.upper(),
            "destinationLocationCode": destination.upper(),
            "departureDate": departure_date,
            "adults": adults,
            "max": 8,
            "currencyCode": "INR",
            "travelClass": "ECONOMY"
        }
        if return_date:
            params["returnDate"] = return_date

        r = requests.get(url, headers=headers, params=params, timeout=20)
        if r.status_code != 200:
            return {"error": f"{r.status_code}: {r.text}"}

        offers = []
        for o in r.json().get("data", []):
            itineraries = o.get("itineraries", [])

            outbound_segments = []
            return_segments = []

            if len(itineraries) >= 1:
                outbound_segments = [
                    {
                        "carrier": s["carrierCode"],
                        "flight": s["number"],
                        "depart": s["departure"]["at"],
                        "arrive": s["arrival"]["at"],
                        "duration": s["duration"]
                    } for s in itineraries[0]["segments"]
                ]

            if len(itineraries) >= 2:
                return_segments = [
                    {
                        "carrier": s["carrierCode"],
                        "flight": s["number"],
                        "depart": s["departure"]["at"],
                        "arrive": s["arrival"]["at"],
                        "duration": s["duration"]
                    } for s in itineraries[1]["segments"]
                ]

            offers.append({
                "price": o["price"]["grandTotal"],
                "currency": o["price"]["currency"],
                "outbound": outbound_segments,
                "return": return_segments
            })

        return {"offers": offers}


class WeatherQueryInput(BaseModel):
    location: str = Field(..., description="City name (e.g. Goa, Ahmedabad)")

class WeatherTool(BaseTool):
    name: str = "weather_lookup"
    description: str = "Get current weather using OpenWeatherMap"
    args_schema: Type[BaseModel] = WeatherQueryInput
    wrapper: OpenWeatherMapAPIWrapper = Field(default_factory=OpenWeatherMapAPIWrapper)

    def _run(self, location: str) -> str:
        try:
            return self.wrapper.run(location)
        except Exception as e:
            return f"Weather fetch failed: {str(e)}"
        
class AccommodationInput(BaseModel):
    location: str = Field(..., description="City name")
    check_in: str = Field(..., description="Check-in date in YYYY-MM-DD")
    check_out: str = Field(..., description="Check-out date in YYYY-MM-DD")

class AccommodationSearchTool(BaseTool):
    name: str = "accommodation_search"
    description: str = "Searches for hotels and stays using Google Serper"
    args_schema: Type[BaseModel] = AccommodationInput
    serper: GoogleSerperAPIWrapper = Field(default_factory=GoogleSerperAPIWrapper)

    def _run(self, location: str, check_in: str, check_out: str) -> str:
        query = f"hotels in {location} from {check_in} to {check_out}"
        try:
            return self.serper.run(query)
        except Exception as e:
            return f"Accommodation search failed: {str(e)}"
        
class TrainSearchInput(BaseModel):
    origin: str = Field(..., description="Departure city")
    destination: str = Field(..., description="Arrival city")
    date: str = Field(..., description="Travel date in YYYY-MM-DD")

class TrainSearchTool(BaseTool):
    name: str = "train_search"
    description: str = "Searches train availability using Google search"
    args_schema: Type[BaseModel] = TrainSearchInput
    serper: GoogleSerperAPIWrapper = Field(default_factory=GoogleSerperAPIWrapper)

    def _run(self, origin: str, destination: str, date: str) -> str:
        query = f"train from {origin} to {destination} on {date}"
        try:
            return self.serper.run(query)
        except Exception as e:
            return f"Train search failed: {str(e)}"

class VisaRequirementsInput(BaseModel):
    nationality: str = Field(..., description="Traveler's nationality")
    destination: str = Field(..., description="Destination country")
    travel_date: str = Field(..., description="Travel date in YYYY-MM-DD format")
    duration_days: int = Field(..., description="Duration of stay in days")

class VisaRequirementsTool(BaseTool):
    name: str = "visa_requirements_search"
    description: str = "Search visa requirements and application process for destinations"
    args_schema: Type[BaseModel] = VisaRequirementsInput
    serper: GoogleSerperAPIWrapper = Field(default_factory=GoogleSerperAPIWrapper)

    def _run(self, nationality: str, destination: str, travel_date: str, duration_days: int) -> str:
        query = f"visa requirements for {nationality} passport holders to {destination} {duration_days} days stay"
        try:
            return self.serper.run(query)
        except Exception as e:
            return f"Visa requirements search failed: {str(e)}"

class LocalExperiencesInput(BaseModel):
    location: str = Field(..., description="City or area name")
    interests: list = Field(..., description="List of traveler interests")
    dates: str = Field(..., description="Travel dates")

class LocalExperiencesTool(BaseTool):
    name: str = "local_experiences_search"
    description: str = "Find local experiences, restaurants, and hidden gems"
    args_schema: Type[BaseModel] = LocalExperiencesInput
    serper: GoogleSerperAPIWrapper = Field(default_factory=GoogleSerperAPIWrapper)

    def _run(self, location: str, interests: list, dates: str) -> str:
        interests_str = ", ".join(interests)
        query = f"unique local experiences {interests_str} in {location} {dates} hidden gems authentic"
        try:
            return self.serper.run(query)
        except Exception as e:
            return f"Local experiences search failed: {str(e)}"

class EventSearchInput(BaseModel):
    location: str = Field(..., description="City or area name")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    interests: list = Field(default=[], description="List of event interests")

class EventSearchTool(BaseTool):
    name: str = "event_search"
    description: str = "Find events, festivals, and activities during travel dates"
    args_schema: Type[BaseModel] = EventSearchInput
    serper: GoogleSerperAPIWrapper = Field(default_factory=GoogleSerperAPIWrapper)

    def _run(self, location: str, start_date: str, end_date: str, interests: list = []) -> str:
        interests_str = " ".join(interests) if interests else ""
        query = f"events festivals {interests_str} in {location} from {start_date} to {end_date}"
        try:
            return self.serper.run(query)
        except Exception as e:
            return f"Event search failed: {str(e)}"

def retry_on_exception(retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:  # Last attempt
                        raise e
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class TravelRequest(BaseModel):
    budget_in_inr: float = Field(..., gt=0)
    travel_by: str = Field(..., pattern="^(flight|train)$")
    travelers: Dict[str, int]
    duration_days: int = Field(..., gt=0)
    departure_date: str
    return_date: str
    departure_city: str
    arrival_city: str
    interests: List[str] = []
    special_requirements: List[str] = []
    travel_style: str = "mid-range"
    accommodation_type: str = "hotel"
    accommodation_budget_per_night: float = Field(..., gt=0)
    nationality: Optional[str] = None

    @field_validator('departure_date', 'return_date')
    @classmethod
    def validate_dates(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Dates must be in YYYY-MM-DD format")

    @field_validator('return_date')
    @classmethod
    def validate_return_after_departure(cls, v: str, info: ValidationInfo) -> str:
        if 'departure_date' in info.data:
            departure = datetime.strptime(info.data['departure_date'], "%Y-%m-%d")
            return_date = datetime.strptime(v, "%Y-%m-%d")
            if return_date < departure:
                raise ValueError("Return date must be after departure date")
        return v

@CrewBase
class TripPlanner:
    """Trip planning crew that coordinates multiple agents to create comprehensive travel plans."""
    
    def __init__(self, llm_manager=None):
        # Call super().__init__() first
        super().__init__()
        
        try:
            # Use provided LLM manager or create new one
            if llm_manager is None:
                self.llm_manager = LLMManager()
            else:
                self.llm_manager = llm_manager
            
            self.llm = self.llm_manager.get_crewai_llm()
            print(f"🤖 Using LLM Provider: {self.llm_manager.get_current_provider()}")
            print(f"🤖 Using LLM Model: {self.llm.model}")
            
            # Load config files from default CrewAI paths
            self.agents_config = {}
            self.tasks_config = {}
            try:
                with open("config/agents.yaml") as f:
                    self.agents_config = yaml.safe_load(f)
                with open("config/tasks.yaml") as f:
                    self.tasks_config = yaml.safe_load(f)
            except FileNotFoundError as e:
                print(f"Warning: Configuration file not found: {e}")
            except yaml.YAMLError as e:
                print(f"Warning: Invalid YAML configuration: {e}")
                
        except Exception as e:
            print(f"Error initializing TripPlanner: {e}")
            raise

    def validate_travel_request(self, travel_request: dict) -> TravelRequest:
        """Validate and convert the travel request dict to a TravelRequest model."""
        return TravelRequest(**travel_request)

    @retry_on_exception(retries=3, delay=1)
    def execute_task(self, task: Task, inputs: dict) -> Any:
        """Execute a task with retry logic."""
        return task.execute(inputs)

    @agent
    def budget_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config["budget_analyzer"], 
            llm=self.llm, 
            tools=[SerperDevTool()],
            verbose=True
        )

    @agent
    def destination_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["destination_researcher"], 
            llm=self.llm, 
            tools=[WeatherTool(),SerperDevTool()], 
            verbose=True
        )

    @agent
    def itinerary_planner(self) -> Agent:
        return Agent(
            config=self.agents_config["itinerary_planner"], 
            llm=self.llm, 
            tools=[SerperDevTool(), WeatherTool()], 
            verbose=True
        )

    @agent
    def accommodation_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["accommodation_specialist"], 
            llm=self.llm, 
            tools = [AccommodationSearchTool()],
            verbose=True
        )

    @agent
    def travel_coordinator(self) -> Agent:
        return Agent(
            config=self.agents_config["travel_coordinator"], 
            llm=self.llm, 
            tools=[AmadeusFlightsTool(),TrainSearchTool(), WeatherTool(), SerperDevTool()], 
            verbose=True
        )

    @agent
    def visa_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config["visa_advisor"],
            llm=self.llm,
            tools=[VisaRequirementsTool(), SerperDevTool()],
            verbose=True
        )

    @agent
    def local_guide(self) -> Agent:
        return Agent(
            config=self.agents_config["local_guide"],
            llm=self.llm,
            tools=[LocalExperiencesTool(), SerperDevTool()],
            verbose=True
        )

    @agent
    def event_finder(self) -> Agent:
        return Agent(
            config=self.agents_config["event_finder"],
            llm=self.llm,
            tools=[EventSearchTool(), SerperDevTool()],
            verbose=True
        )
    
    @task
    def budget_analysis_task(self) -> Task:
        return Task(config=self.tasks_config["budget_analysis"])

    @task
    def destination_research_task(self) -> Task:
        return Task(config=self.tasks_config["destination_research"])
    
    @task
    def itinerary_planning_task(self) -> Task:
        return Task(config=self.tasks_config["itinerary_planning"])
    
    @task
    def accommodation_task(self) -> Task:
        return Task(config=self.tasks_config["accommodation"])
    
    @task
    def travel_coordination_task(self) -> Task:
        return Task(config=self.tasks_config["travel_coordination"])

    @task
    def visa_advice_task(self) -> Task:
        return Task(config=self.tasks_config["visa_advice"])

    @task
    def local_experiences_task(self) -> Task:
        return Task(config=self.tasks_config["local_experiences"])

    @task
    def event_finding_task(self) -> Task:
        return Task(config=self.tasks_config["event_finding"])
    
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.budget_analyzer(),
                self.destination_researcher(),
                self.itinerary_planner(),
                self.accommodation_specialist(),
                self.travel_coordinator(),
                self.visa_advisor(),
                self.local_guide(),
                self.event_finder()
            ],
            tasks=[
                self.budget_analysis_task(),
                self.destination_research_task(),
                self.itinerary_planning_task(),
                self.accommodation_task(),
                self.travel_coordination_task(),
                self.visa_advice_task(),
                self.local_experiences_task(),
                self.event_finding_task()
            ],
            process=Process.sequential,
            verbose=True
        )

    def kickoff(self, inputs: dict) -> Any:
        """
        Execute the travel planning workflow with input validation and error handling.
        """
        try:
            # Validate travel request
            travel_request = self.validate_travel_request(inputs.get("travel_request", {}))
            
            # Create crew instance
            crew_instance = self.crew()
            
            # Execute tasks with retry logic
            result = crew_instance.kickoff(inputs={"travel_request": travel_request.dict()})
            
            return result
            
        except ValueError as e:
            return {"error": f"Invalid input: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}