#!/usr/bin/env python
import sys
import warnings

from latest_ai_news.crew import LatestAiNews

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    inputs = {
        'topic': 'AI LLMs'
    }
    LatestAiNews().crew().kickoff(inputs=inputs)

