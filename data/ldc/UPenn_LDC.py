import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import time
import sys
import os
import json

# Target page URL
URL = "https://catalog.ldc.upenn.edu/byyear"

# Configuration
YEARS_TO_PROCESS = None  # None to process all years; or a list like [2025, 2024]
MAX_DATASETS = None      # Maximum number of datasets to process; None = no limit
DELAY_BETWEEN_REQUESTS = 1  # Seconds between requests, to avoid being rate-limited

# Resume configuration
PROGRESS_FILE = "ldc_progress.json"
METADATA_FILE = "ldc_datasets_metadata.csv"
ENABLE_RESUME = True  # Whether to enable resume from last saved progress

def save_progress(all_datasets_info, processed_indices, all_metadata):
    """Save progress to file."""
    progress_data = {
        'total_datasets': len(all_datasets_info),
        'processed_indices': processed_indices,
        'processed_count': len(processed_indices),
        'metadata': all_metadata,
        'timestamp': time.time()
    }

    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)

def load_progress():
    """Load previous progress, if any."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load progress file: {e}")
    return None

def extract_dataset_metadata(dataset_url, max_retries=3):
    """
    Extract metadata from a single dataset page, with retry on transient errors.
    """
    for attempt in range(max_retries):
        try:
            # Back off before retrying to avoid hammering the server
            if attempt > 0:
                time.sleep(2 ** attempt)  # exponential backoff

            page = requests.get(dataset_url, timeout=30)
            page.raise_for_status()
            soup = BeautifulSoup(page.content, "html.parser")

            # Locate the metadata table
            metadata = {}

            # Iterate over rows containing metadata fields
            for row in soup.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 2:
                    # First cell is the field name, second is the value
                    field_name = cells[0].get_text(strip=True).rstrip(':')
                    field_value = cells[1].get_text(strip=True)

                    # Special handling for individual fields
                    if field_name == "Item Name":
                        # Title lives inside a <span>
                        title_span = cells[1].find("span")
                        if title_span:
                            metadata["Title"] = title_span.get_text(strip=True)
                    elif field_name in [
                        # Core metadata fields
                        "Author(s)", "LDC Catalog No.", "ISLRN", "DOI", "Release Date",
                        "Member Year(s)", "DCMI Type(s)", "Data Source(s)", "Application(s)",
                        "Language(s)", "Language ID(s)", "License(s)", "Online Documentation",
                        "Licensing Instructions", "Citation",
                        # Optional fields
                        "ISBN", "Sample Type", "Sample Rate", "Project(s)", "Related Works"
                    ]:
                        metadata[field_name] = field_value

            # Special case: Related Works may contain multiple links
            related_works_element = soup.find("td", string="Related Works:")
            if related_works_element:
                next_td = related_works_element.find_next_sibling("td")
                if next_td:
                    related_links = next_td.find_all("a")
                    if related_links:
                        related_works = []
                        for link in related_links:
                            if link.get_text(strip=True) not in ["View", "Hide"]:
                                related_works.append(link.get_text(strip=True))
                        if related_works:
                            metadata["Related Works"] = "; ".join(related_works)

            # Year was already extracted from the listing page; no need to re-extract here

            return metadata

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Failed to fetch {dataset_url} after {max_retries} attempts: {e}")
                return None
            else:
                print(f"Attempt {attempt + 1} failed for {dataset_url}, retrying...")
                continue
        except Exception as e:
            print(f"Error extracting metadata from {dataset_url}: {e}")
            return None

    return None

try:
    # Check whether existing progress is available
    start_index = 0
    existing_metadata = []

    if ENABLE_RESUME:
        existing_progress = load_progress()
        if existing_progress:
            print("Found existing progress file.")
            print(f"Already processed: {existing_progress['processed_count']}/{existing_progress['total_datasets']}")
            print(f"Last saved at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(existing_progress['timestamp']))}")

            choice = input("Resume from previous progress? (y/n): ").lower().strip()
            if choice == 'y':
                print("Resuming from previous progress...")
                start_index = existing_progress['processed_count']
                existing_metadata = existing_progress.get('metadata', [])
                print(f"Starting from dataset #{start_index + 1}...")
            else:
                print("Starting from scratch...")
                if os.path.exists(PROGRESS_FILE):
                    os.remove(PROGRESS_FILE)

    # Fetch the listing page
    page = requests.get(URL)
    # Raise on HTTP errors (e.g. 404 or 500)
    page.raise_for_status()

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(page.content, "html.parser")

    # Each <h2> tag on the page denotes a year
    year_elements = soup.find_all("h2")

    # Collect dataset info and URLs
    all_datasets_info = []
    total_datasets = 0

    # Iterate over each year
    for year_element in year_elements:
        # Get the year label and trim whitespace
        year = year_element.get_text(strip=True)

        # Skip years not in the configured filter
        if YEARS_TO_PROCESS and year not in YEARS_TO_PROCESS:
            continue

        # The first <table> sibling after <h2> is the dataset list for that year
        dataset_table = year_element.find_next_sibling("table")

        # If a dataset table was found
        if dataset_table:
            # Iterate over all <tr> rows
            for row in dataset_table.find_all("tr"):
                # Each row contains the dataset link and title
                cells = row.find_all("td")
                if len(cells) >= 3:
                    # Dataset link is in the first <td>
                    link_element = cells[0].find("a")
                    if link_element and 'href' in link_element.attrs:
                        dataset_url = urljoin(URL, link_element['href'])
                        catalog_id = link_element.get_text(strip=True)

                        # Dataset title is in the third <td>
                        title_span = cells[2].find("span")
                        title = title_span.get_text(strip=True) if title_span else "Unknown"

                        # Collect basic info
                        dataset_info = {
                            'Year': year,
                            'Catalog_ID': catalog_id,
                            'Title': title,
                            'URL': dataset_url
                        }

                        all_datasets_info.append(dataset_info)
                        total_datasets += 1

                        # Stop early once MAX_DATASETS is reached
                        if MAX_DATASETS and total_datasets >= MAX_DATASETS:
                            print(f"Reached MAX_DATASETS cap: {MAX_DATASETS}")
                            break

            # Break out of the year loop too if the cap was hit
            if MAX_DATASETS and total_datasets >= MAX_DATASETS:
                break

    print(f"Found {total_datasets} datasets; collecting metadata...")
    print("This may take a while; please be patient.")

    # Collect metadata for each dataset
    all_metadata = existing_metadata.copy()
    success_count = len([m for m in all_metadata if 'Title' in m and m.get('Title') != 'Unknown'])
    error_count = len(all_metadata) - success_count
    processed_indices = list(range(start_index))

    for i in range(start_index, total_datasets):
        dataset_info = all_datasets_info[i]

        # Print progress
        percent = ((i + 1) / total_datasets) * 100
        print(f"\rProgress: [{i+1:4d}/{total_datasets:4d}] ({percent:5.1f}%) - processing: {dataset_info['Title'][:50]}...")

        metadata = extract_dataset_metadata(dataset_info['URL'])

        if metadata:
            # Merge basic info with extracted metadata
            complete_info = {**dataset_info, **metadata}
            if i < len(all_metadata):
                all_metadata[i] = complete_info
            else:
                all_metadata.append(complete_info)
            success_count += 1
        else:
            # If extraction failed, keep at least the basic info
            if i < len(all_metadata):
                all_metadata[i] = dataset_info
            else:
                all_metadata.append(dataset_info)
            error_count += 1

        processed_indices.append(i)

        # Save progress periodically (every 10 datasets)
        if (i + 1) % 10 == 0:
            save_progress(all_datasets_info, processed_indices, all_metadata)
            print(f"\rProgress: [{i+1:4d}/{total_datasets:4d}] ({percent:5.1f}%) - progress saved")

        # Throttle between requests
        if i < total_datasets - 1 and DELAY_BETWEEN_REQUESTS > 0:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\rProgress: [{total_datasets:4d}/{total_datasets:4d}] (100.0%)")
    print(f"\nMetadata collection done: {success_count} succeeded, {error_count} failed")

    # Final save
    if ENABLE_RESUME:
        save_progress(all_datasets_info, list(range(total_datasets)), all_metadata)

    # Build DataFrame and write CSV
    if all_metadata:
        df = pd.DataFrame(all_metadata)

        # Reorder columns so the most important fields come first
        columns_order = ['Year', 'Catalog_ID', 'Title', 'Author(s)', 'LDC Catalog No.', 'ISBN',
                        'ISLRN', 'DOI', 'Release Date', 'Member Year(s)', 'DCMI Type(s)',
                        'Sample Type', 'Sample Rate', 'Data Source(s)', 'Project(s)',
                        'Application(s)', 'Language(s)', 'Language ID(s)', 'License(s)',
                        'Online Documentation', 'Licensing Instructions', 'Citation',
                        'Related Works', 'URL']

        # Only keep columns that actually exist
        existing_columns = [col for col in columns_order if col in df.columns]
        df = df[existing_columns]

        # Save to CSV
        output_file = METADATA_FILE
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nMetadata table saved to: {output_file}")
        print(f"Collected metadata for {len(all_metadata)} datasets")

        # Preview the first few rows
        print("\n--- Preview ---")
        print(df.head())

        # Print summary statistics
        print("\n--- Statistics ---")
        print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
        print(f"Languages: {df['Language(s)'].nunique()} unique")
        print(f"Projects: {df['Project(s)'].nunique()} unique")

    else:
        print("No metadata could be collected")


except requests.exceptions.RequestException as e:
    print(f"Error: cannot reach URL. Check your network connection. ({e})")
except Exception as e:
    print(f"Unexpected error during processing: {e}")
