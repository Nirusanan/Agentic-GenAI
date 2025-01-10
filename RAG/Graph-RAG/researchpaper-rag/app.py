import os
import pandas as pd
import tiktoken
import asyncio
from rich import print
from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports, read_indexer_communities
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch

# LLM 
api_key = os.environ["OPENAI_API_KEY"]
llm_model = "gpt-4o-mini"
llm = ChatOpenAI(
    api_key=api_key,
    model=llm_model,
    api_type=OpenaiApiType.OpenAI,
    max_retries=5,
)
token_encoder = tiktoken.get_encoding("cl100k_base")

# Load Context
INPUT_DIR = "output" # After indexing, output folder contains graphml and parquet files. Those are inputs for the context
COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
COMMUNITY_TABLE = "create_final_communities"

COMMUNITY_LEVEL = 0
entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
community_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_TABLE}.parquet")

communities = read_indexer_communities(community_df, entity_df, report_df)
reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
print(f"Report records: {len(report_df)}")
report_df.head()

context_builder = GlobalCommunityContext(
    community_reports=reports,
    entities=entities,
    communities=communities,
    token_encoder=token_encoder,
)

# Setup Global Search - wide range response
context_builder_params = { "use_community_summary": False, "shuffle_data": True, "include_community_rank": True, "min_community_rank": 0, "community_rank_name": "rank", "include_community_weight": True, "community_weight_name": "occurrence weight", "normalize_community_weight": True, "max_tokens": 3_000, "context_name": "Reports", }
map_llm_params = { "max_tokens": 1000, "temperature": 0.0, "response_format": {"type": "json_object"}, }
reduce_llm_params = { "max_tokens": 2000, "temperature": 0.0, }

search_engine = GlobalSearch(
    llm=llm,
    context_builder=context_builder,
    token_encoder=token_encoder,
    max_data_tokens=12_000,
    map_llm_params=map_llm_params,
    reduce_llm_params=reduce_llm_params,
    allow_general_knowledge=False,
    json_mode=True,
    context_builder_params=context_builder_params,
    concurrent_coroutines=10,
    response_type="multiple-page report", 
)

# Run Global Search
async def main(query: str):
    result = await search_engine.asearch(query)
    return result

if __name__ == "__main__":
    query = "What are the LLM future works?"
    result = asyncio.run(main(query))
    print(result.response)
    print(result.context_data["reports"])
    print(f"LLM calls: {result.llm_calls}. LLM tokens: {result.prompt_tokens}")