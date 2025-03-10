import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Optional, List, Dict
import json

class DatasetSearcher:
    def __init__(self):
        self.recent_results = pd.DataFrame()
        self.cache_dir = "dataset_search_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "recent_results.csv")

    def search_kaggle(self, query: str, max_results: int = 5) -> pd.DataFrame:
        """
        Search for datasets on Kaggle
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            DataFrame containing search results
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            kaggle_api = KaggleApi()
            kaggle_api.authenticate()
            
            datasets = kaggle_api.dataset_list(search=query)
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
        except Exception as e:
            print(f"Error searching Kaggle: {e}")
            return pd.DataFrame(columns=["Source", "Title", "Description", "Size", "Format", "Download Link"])

    def search_google_dataset_search(self, query: str, max_results: int = 5) -> pd.DataFrame:
        """
        Search for datasets using Google Dataset Search
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            DataFrame containing search results
        """
        try:
            url = f"https://datasetsearch.research.google.com/search?query={query}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            
            results = []
            dataset_elements = soup.select(".dataset-card") or soup.select("[data-testid='dataset-item']") or []
            
            for item in dataset_elements[:max_results]:
                try:
                    title_element = item.select_one(".dataset-title") or item.select_one("h2")
                    desc_element = item.select_one(".dataset-description") or item.select_one("p")
                    link_element = item.select_one("a")
                    
                    title = title_element.text.strip() if title_element else "Unknown Title"
                    description = desc_element.text.strip() if desc_element else "No description available"
                    link = link_element["href"] if link_element and link_element.has_attr("href") else "#"
                    
                    if not link.startswith("http"):
                        link = f"https://datasetsearch.research.google.com{link}"
                    
                    results.append({
                        "Source": "Google Dataset Search",
                        "Title": title,
                        "Description": description[:200] + "..." if len(description) > 200 else description,
                        "Size": "Unknown",
                        "Format": "Unknown",
                        "Download Link": link
                    })
                except Exception as e:
                    print(f"Error parsing dataset element: {e}")
                    continue
                    
            return pd.DataFrame(results)
        except Exception as e:
            print(f"Error searching Google Dataset Search: {e}")
            return pd.DataFrame(columns=["Source", "Title", "Description", "Size", "Format", "Download Link"])

    def search_huggingface(self, query: str, max_results: int = 5) -> pd.DataFrame:
        """
        Search for datasets on Hugging Face Hub
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            DataFrame containing search results
        """
        try:
            api_url = f"https://huggingface.co/api/datasets?search={query}&limit={max_results}"
            response = requests.get(api_url)
            
            if response.status_code == 200:
                datasets = response.json()
                results = []
                
                for dataset in datasets:
                    results.append({
                        "Source": "Hugging Face",
                        "Title": dataset.get("id", "Unknown"),
                        "Description": dataset.get("description", "No description")[:200] + "..." if dataset.get("description", "") and len(dataset.get("description", "")) > 200 else dataset.get("description", "No description"),
                        "Size": dataset.get("downloads", "Unknown"),
                        "Format": dataset.get("card_data", {}).get("tags", ["Unknown"])[0] if dataset.get("card_data", {}).get("tags", []) else "Unknown",
                        "Download Link": f"https://huggingface.co/datasets/{dataset.get('id', '')}"
                    })
                    
                return pd.DataFrame(results)
            else:
                print(f"Error searching Hugging Face: {response.status_code}")
                return pd.DataFrame(columns=["Source", "Title", "Description", "Size", "Format", "Download Link"])
        except Exception as e:
            print(f"Error searching Hugging Face: {e}")
            return pd.DataFrame(columns=["Source", "Title", "Description", "Size", "Format", "Download Link"])

    def search_datasets(self, query: str, sources: List[str] = None, max_results: int = 5) -> pd.DataFrame:
        """
        Search for datasets across multiple sources
        
        Args:
            query: Search query string
            sources: List of sources to search (default: all available sources)
            max_results: Maximum number of results per source
            
        Returns:
            DataFrame containing combined search results
        """
        all_sources = ["kaggle", "google", "huggingface"]
        sources = sources or all_sources
        results = []
        
        if "kaggle" in sources:
            kaggle_results = self.search_kaggle(query, max_results)
            results.append(kaggle_results)
            
        if "google" in sources:
            google_results = self.search_google_dataset_search(query, max_results)
            results.append(google_results)
            
        if "huggingface" in sources:
            hf_results = self.search_huggingface(query, max_results)
            results.append(hf_results)
            
        # Combine and reset index
        combined_results = pd.concat(results, ignore_index=True) if results else pd.DataFrame(
            columns=["Source", "Title", "Description", "Size", "Format", "Download Link"]
        )
        
        # Cache the results
        self.recent_results = combined_results
        try:
            self.recent_results.to_csv(self.cache_file, index=False)
        except Exception as e:
            print(f"Error caching search results: {e}")
        
        return combined_results

    def get_recent_results(self) -> pd.DataFrame:
        """
        Get the most recent search results
        
        Returns:
            DataFrame containing the most recent search results
        """
        if not self.recent_results.empty:
            return self.recent_results
        
        # Try to load from cache
        try:
            if os.path.exists(self.cache_file):
                return pd.read_csv(self.cache_file)
        except Exception as e:
            print(f"Error loading cached results: {e}")
            
        return pd.DataFrame(columns=["Source", "Title", "Description", "Size", "Format", "Download Link"])

    def download_dataset(self, url: str, save_path: str = "datasets") -> str:
        """
        Download a dataset from a URL
        
        Args:
            url: URL to download from
            save_path: Path to save the dataset to
            
        Returns:
            Path to the downloaded dataset
        """
        os.makedirs(save_path, exist_ok=True)
        dataset_name = url.split("/")[-1] if "/" in url else "dataset"
        output_dir = os.path.join(save_path, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        if "kaggle.com" in url:
            return self._download_kaggle_dataset(url, output_dir)
        elif "huggingface.co" in url:
            return self._download_huggingface_dataset(url, output_dir)
        else:
            return self._download_generic_dataset(url, output_dir)

    def _download_kaggle_dataset(self, url: str, output_dir: str) -> str:
        """
        Download a dataset from Kaggle
        
        Args:
            url: Kaggle dataset URL
            output_dir: Directory to save the dataset to
            
        Returns:
            Path to the downloaded dataset
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            kaggle_api = KaggleApi()
            kaggle_api.authenticate()
            
            # Parse dataset reference from URL
            parts = url.split("/")
            if "datasets" in parts:
                datasets_index = parts.index("datasets")
                if datasets_index + 2 < len(parts):
                    owner = parts[datasets_index + 1]
                    dataset_name = parts[datasets_index + 2]
                    dataset_ref = f"{owner}/{dataset_name}"
                    print(f"Downloading Kaggle dataset: {dataset_ref}")
                    kaggle_api.dataset_download_files(dataset_ref, path=output_dir, unzip=True)
                    return output_dir
                else:
                    raise ValueError("Could not parse dataset reference from URL")
            else:
                # Try to parse in a different way
                if len(parts) >= 2:
                    dataset_ref = "/".join(parts[-2:])
                    print(f"Downloading Kaggle dataset: {dataset_ref}")
                    kaggle_api.dataset_download_files(dataset_ref, path=output_dir, unzip=True)
                    return output_dir
                else:
                    raise ValueError("Could not parse dataset reference from URL")
        except Exception as e:
            print(f"Error downloading Kaggle dataset: {e}")
            print("Falling back to generic download method")
            return self._download_generic_dataset(url, output_dir)

    def _download_huggingface_dataset(self, url: str, output_dir: str) -> str:
        """
        Download a dataset from Hugging Face
        
        Args:
            url: Hugging Face dataset URL
            output_dir: Directory to save the dataset to
            
        Returns:
            Path to the downloaded dataset
        """
        try:
            from datasets import load_dataset
            
            # Parse dataset ID from URL
            parts = url.split("/")
            if "datasets" in parts:
                dataset_index = parts.index("datasets")
                if dataset_index + 1 < len(parts):
                    dataset_id = parts[dataset_index + 1]
                    
                    # Load and save the dataset
                    print(f"Downloading Hugging Face dataset: {dataset_id}")
                    dataset = load_dataset(dataset_id)
                    
                    # Save each split
                    for split_name, split_data in dataset.items():
                        split_path = os.path.join(output_dir, f"{split_name}.csv")
                        split_data.to_csv(split_path)
                    
                    # Save dataset info
                    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
                        json.dump({
                            "dataset_id": dataset_id,
                            "splits": list(dataset.keys()),
                            "source": "Hugging Face"
                        }, f, indent=2)
                    
                    return output_dir
                else:
                    raise ValueError("Could not parse dataset ID from URL")
            else:
                raise ValueError("Not a valid Hugging Face dataset URL")
        except Exception as e:
            print(f"Error downloading Hugging Face dataset: {e}")
            print("Falling back to generic download method")
            return self._download_generic_dataset(url, output_dir)

    def _download_generic_dataset(self, url: str, output_dir: str) -> str:
        """
        Download a dataset from a generic URL
        
        Args:
            url: URL to download from
            output_dir: Directory to save the dataset to
            
        Returns:
            Path to the downloaded dataset
        """
        try:
            print(f"Downloading dataset from: {url}")
            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                # Try to get filename from content-disposition header
                content_disposition = response.headers.get("content-disposition")
                if content_disposition and "filename=" in content_disposition:
                    filename = content_disposition.split("filename=")[1].strip('"\'')
                else:
                    filename = url.split("/")[-1]
                    if not filename or "?" in filename:
                        filename = "dataset.zip"
                
                file_path = os.path.join(output_dir, filename)
                
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 KB
                
                with open(file_path, "wb") as f:
                    for data in response.iter_content(block_size):
                        f.write(data)
                
                print(f"Downloaded {file_path}")
                
                # Try to extract if it's a compressed file
                if filename.endswith((".zip", ".tar.gz", ".tgz")):
                    try:
                        self._extract_archive(file_path, output_dir)
                    except Exception as e:
                        print(f"Error extracting archive: {e}")
                
                return output_dir
            else:
                raise Exception(f"Failed to download. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            with open(os.path.join(output_dir, "download_error.txt"), "w") as f:
                f.write(f"Error downloading dataset from {url}: {str(e)}")
            return output_dir

    def _extract_archive(self, archive_path: str, extract_path: str) -> None:
        """
        Extract an archive file
        
        Args:
            archive_path: Path to the archive file
            extract_path: Path to extract to
        """
        import zipfile
        import tarfile
        import shutil
        
        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"Extracted zip file to {extract_path}")
        elif archive_path.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as tar_ref:
                tar_ref.extractall(extract_path)
            print(f"Extracted tar.gz file to {extract_path}")
        
        # Clean up the archive file
        os.remove(archive_path)

if __name__ == "__main__":
    # Test the DatasetSearcher
    searcher = DatasetSearcher()
    
    query = "climate change"
    print(f"Searching for datasets related to '{query}'...")
    
    results = searcher.search_datasets(query)
    print("\nSearch Results:")
    print(results)
    
    if not results.empty:
        try:
            choice = int(input("\nEnter the row number of the dataset you want to download: "))
            selected_dataset = results.iloc[choice]
            print(f"\nDownloading dataset: {selected_dataset['Title']}")
            output_path = searcher.download_dataset(selected_dataset["Download Link"])
            print(f"Dataset downloaded to: {output_path}")
        except (ValueError, IndexError):
            print("Invalid selection. Please try again.")
    else:
        print("No datasets found.")
