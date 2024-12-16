# Crew Agent for LatestAiNews 

## Installation

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```
### Customizing

**Add your `GROQ_API_KEY` and `SERPER_API_KEY` into the `.env` file**

- Modify `src/latest_ai_news/config/agents.yaml` to define your agents
- Modify `src/latest_ai_news/config/tasks.yaml` to define your tasks
- Modify `src/latest_ai_news/crew.py` to add your own logic, tools and specific args
- Modify `src/latest_ai_news/main.py` to add custom inputs for your agents and tasks

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
$ crewai run
```

This command initializes the latest-ai-news Crew, assembling the agents and assigning them tasks as defined in your configuration.

This example, unmodified, will run the create a `report.md` file with the output of a research on LLMs in the root folder.

## Understanding Your Crew

The latest-ai-news Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.


