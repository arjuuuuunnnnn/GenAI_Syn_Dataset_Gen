import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

class DatasetSearchAgent:
    def __init__(self):
        self.kaggle_api = KaggleApi()
        kaggle_json_path = os.path.expanduser("kaggle.json")
        if not os.path.exists(kaggle_json_path):
            raise FileNotFoundError(f"kaggle.json not found at {kaggle_json_path}")
        self.kaggle_api.authenticate()

    def search_kaggle(self, query: str, max_results: int = 5) -> pd.DataFrame:
        datasets = self.kaggle_api.dataset_list(search=query)
        results = []
        for dataset in datasets[:max_results]:
            results.append({
                "Source": "Kaggle",
                "Title": dataset.title,
                "Description": dataset.description[:200] + "..." if len(dataset.description) > 200 else dataset.description,
                "Size": dataset.size,
                "Format": ", ".join(dataset.fileTypes) if hasattr(dataset, 'fileTypes') and dataset.fileTypes else "Unknown",
                "Download Link": f"https://www.kaggle.com/{dataset.ref}"
            })
        return pd.DataFrame(results)

    def search_google_dataset_search(self, query: str, max_results: int = 5) -> pd.DataFrame:
        url = f"https://datasetsearch.research.google.com/search?query={query}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for item in soup.select(".dataset-item")[:max_results]:
            title = item.select_one(".dataset-title").text.strip()
            description = item.select_one(".dataset-description").text.strip()
            link = item.select_one(".dataset-link")["href"]
            results.append({
                "Source": "Google Dataset Search",
                "Title": title,
                "Description": description,
                "Download Link": link
            })
        return pd.DataFrame(results)

    def search_datasets(self, query: str, max_results: int = 5) -> pd.DataFrame:
        kaggle_results = self.search_kaggle(query, max_results)
        google_results = self.search_google_dataset_search(query, max_results)
        return pd.concat([kaggle_results, google_results], ignore_index=True)

    def download_dataset(self, url: str, save_path: str = "datasets"):
        os.makedirs(save_path, exist_ok=True)
        if "kaggle.com" in url:
            parts = url.split("/")
            if "datasets" in parts:
                datasets_index = parts.index("datasets")
                if datasets_index + 2 < len(parts):
                    owner = parts[datasets_index + 1]
                    dataset_name = parts[datasets_index + 2]
                    dataset_ref = f"{owner}/{dataset_name}"
                    print(f"Dataset URL: {url}")
                    print(f"Parsed reference: {dataset_ref}")
                    try:
                        self.kaggle_api.dataset_download_files(dataset_ref, path=save_path, unzip=True)
                        print(f"Dataset downloaded to {save_path}")
                    except Exception as e:
                        print(f"Error downloading dataset: {e}")
                        print("\nTry downloading manually from the URL.")
                else:
                    print("Could not parse dataset reference from URL.")
            else:
                try:
                    if len(parts) >= 2:
                        dataset_ref = "/".join(parts[-2:])
                        print(f"Dataset URL: {url}")
                        print(f"Parsed reference: {dataset_ref}")
                        self.kaggle_api.dataset_download_files(dataset_ref, path=save_path, unzip=True)
                        print(f"Dataset downloaded to {save_path}")
                    else:
                        print("Could not parse dataset reference from URL.")
                except Exception as e:
                    print(f"Error downloading dataset: {e}")
                    print("\nTry downloading manually from the URL.")
        else:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    filename = os.path.join(save_path, url.split("/")[-1])
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    print(f"Dataset downloaded to {filename}")
                else:
                    print(f"Failed to download dataset. Status code: {response.status_code}")
            except Exception as e:
                print(f"Error downloading dataset: {e}") 

if __name__ == "__main__":
    agent = DatasetSearchAgent()

    # User query
    query = "climate change"
    print(f"Searching for datasets related to '{query}'...")

    results = agent.search_datasets(query)
    print("\nSearch Results:")
    print(results)

    if not results.empty:
        try:
            choice = int(input("\nEnter the row number of the dataset you want to download: "))
            selected_dataset = results.iloc[choice]
            print(f"\nDownloading dataset: {selected_dataset['Title']}")
            agent.download_dataset(selected_dataset["Download Link"])
        except (ValueError, IndexError):
            print("Invalid selection. Please try again.")
    else:
        print("No datasets found.")
