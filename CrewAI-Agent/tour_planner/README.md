# Crew Agent for Tour Planning
## Introduction
This project is an example using the CrewAI framework to automate the process of planning a trip if you are in doubt between different options. CrewAI orchestrates autonomous AI agents, enabling them to collaborate and execute complex tasks efficiently.

## CrewAI Framework
CrewAI is designed to facilitate the collaboration of role-playing AI agents. In this example, these agents work together to choose between different of cities and put together a full itinerary for the trip based on your preferences.


- **Configure Environment**: Create ``.env` and set up the environment variables for [Browseless](https://www.browserless.io/), [Serper](https://serper.dev/) and [OpenAI](https://platform.openai.com/api-keys)
- **Install Dependencies**: Run `poetry install --no-root`.
- **Execute the Script**: Run `poetry run python main.py` and input your idea.

## Details & Explanation
- **Running the Script**: Execute `python main.py`` and input your idea when prompted. The script will leverage the CrewAI framework to process the idea and generate a landing page.
- **Key Components**:
  - `./main.py`: Main script file.
  - `./trip_tasks.py`: Main file with the tasks prompts.
  - `./trip_agents.py`: Main file with the agents creation.
  - `./tools`: Contains tool classes used by the agents.



