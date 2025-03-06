from serpapi.google_search import GoogleSearch
import pandas as pd
import json

class SearchAgent:
    def __init__(self, api_key):
        self.api_key = api_key

    def process_query(self, query):
        processed_query = " ".join(query.split()).lower()
        return processed_query

    def search_web(self, query):
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": 5,
            "google_domain": "google.com",
            "hl": "en",
            "gl": "us"
        }
        try:
            search = GoogleSearch(params)
            search_results = search.get_dict()
            return self.extract_results(search_results)
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def extract_results(self, search_results):
        results = []
        for result in search_results.get("organic_results", []):
            extracted_result = {
                "title": result.get("title"),
                "link": result.get("link"),
                "snippet": result.get("snippet")
            }
            results.append(extracted_result)
        return results

    def save_to_csv(self, data, filename="search_results.csv"):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Search results saved to {filename}")

    def save_to_json(self, data, filename="search_results.json"):
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Search results saved to {filename}")

    def run(self, user_query, output_format="csv"):
        processed_query = self.process_query(user_query)
        raw_web_data = self.search_web(processed_query)

        if output_format == "csv":
            self.save_to_csv(raw_web_data)
        elif output_format == "json":
            self.save_to_json(raw_web_data)
        else:
            print("Invalid output format. Supported formats: 'csv', 'json'")

        return raw_web_data
