import json
import os
import pandas as pd


from search_agent import SearchAgent

API_KEY = "16bd3521f41594d6072fa4b4b27a45a0c63e434427cc313d40d6233db8f1f936"


search_agent = SearchAgent(API_KEY)


user_query = input("Enter your search query: ")


raw_web_data = search_agent.run(user_query, output_format="csv")


print(json.dumps(raw_web_data, indent=2))
