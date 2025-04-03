import requests
import re
import tarfile
import os
import xml.etree.ElementTree as ET
import urllib.parse
import tempfile
from fuzzywuzzy import fuzz
import time
import torch
from torch_geometric.data import Data
import random
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
import backoff
from scholarly import scholarly, ProxyGenerator


USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
]

def setup_proxy():
    try:
        pg = ProxyGenerator()
        success = pg.FreeProxies()  # or pg.Tor_Internal()
        
        if success:
            scholarly.use_proxy(pg)
            print("Successfully set up proxy for scholarly.")
            return pg
        else:
            print("Failed to set up proxy.")
            return None
            
    except ImportError:
        print("Could not import scholarly. Please install it with 'pip install scholarly'.")
        return None
    except Exception as e:
        print(f"Error setting up proxy: {str(e)}")
        return None

def clean_title(title):
    return re.sub(r'\W+', ' ', title).strip().lower()

@backoff.on_exception(backoff.expo, RequestException, max_tries=3)
def get_arxiv_id(title, max_results=10, similarity_threshold=80):
    cleaned_title = clean_title(title)
    words = cleaned_title.split()

    # Construct query with all words in the title
    query = '+AND+'.join(f'all:{urllib.parse.quote(word)}' for word in words)

    base_url = "http://export.arxiv.org/api/query?"
    search_query = f'{base_url}search_query={query}&start=0&max_results={max_results}'

    response = requests.get(search_query)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}

    best_match = None
    highest_similarity = 0

    for entry in root.findall('atom:entry', namespace):
        entry_title = entry.find('atom:title', namespace).text
        similarity = fuzz.ratio(cleaned_title, clean_title(entry_title))

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = entry

        if similarity == 100:  # Exact match found
            break

    if best_match is not None and highest_similarity >= similarity_threshold:
        arxiv_url = best_match.find('atom:id', namespace).text
        arxiv_id = arxiv_url.split('/')[-1]
        return arxiv_id
    else:
        return None

def get_arxiv_ids(titles):
    arxiv_ids = []
    for title in titles:
        try:
            arxiv_id = get_arxiv_id(title)
            arxiv_ids.append(arxiv_id)
            time.sleep(0.5)
        except Exception as e:
            print(f"Error finding ArXiv ID for '{title}': {str(e)}")
            arxiv_ids.append(None)
    return arxiv_ids

@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=5)
def download_arxiv_source(arxiv_id):
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as temp_file:
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        return temp_file.name

def find_main_tex_file(directory):
    tex_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".tex"):
                tex_files.append(os.path.join(root, file))

    if not tex_files:
        return None

    main_file_candidates = [f for f in tex_files if re.match(r'(main|paper|article)\.tex', os.path.basename(f), re.IGNORECASE)]
    if main_file_candidates:
        return main_file_candidates[0]

    for file in tex_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                if '\\documentclass' in content and '\\begin{document}' in content:
                    return file
        except UnicodeDecodeError:
            try:
                with open(file, 'r', encoding='latin-1') as f:
                    content = f.read()
                    if '\\documentclass' in content and '\\begin{document}' in content:
                        return file
            except Exception:
                print(f"Unable to read file: {file}")

    return max(tex_files, key=os.path.getsize)

def process_input_commands(content, base_dir):
    def replace_input(match):
        input_file = match.group(1)
        if not input_file.endswith('.tex'):
            input_file += '.tex'
        input_path = os.path.join(base_dir, input_file)
        if os.path.exists(input_path):
            try:
                with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return process_input_commands(f.read(), os.path.dirname(input_path))
            except Exception as e:
                print(f"Error processing input file {input_path}: {str(e)}")
                return ''
        return ''

    processed_content = re.sub(r'\\input\{([^}]+)\}', replace_input, content)
    return processed_content

def find_reference_files(tex_content):
    ref_commands = re.findall(r'\\(?:bibliography|addbibresource|bibdata){([^}]+)\}', tex_content)
    ref_files = []
    for command in ref_commands:
        ref_files.extend(command.split(','))
    return [f"{ref.strip()}" for ref in ref_files]

def parse_bib_content(bib_content):
    entries = re.split(r'(@\w+\s*\{[^@]*?\n\})', bib_content, flags=re.DOTALL)
    titles = []

    for entry in entries:
        if entry.strip().startswith('@'):
            # Extract the title from valid entries
            title_match = re.search(r'title\s*=\s*["{](.+?)["}]', entry, re.DOTALL | re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()
                title = re.sub(r'\s+', ' ', title)
                titles.append(title)

    return titles

def parse_bbl_content(bbl_content):
    entries = re.split(r'\\bibitem', bbl_content)[1:]
    titles = []

    for entry in entries:
        entry = re.sub(r'\s+', ' ', entry).strip()
        title = None

        title_patterns = [
            r'\\newblock\s*(.*?)\s*\\newblock',  # Captures content between first two \newblock commands
            r'{\\em\s+(.*?)}',
            r'``(.*?)\'\'',
            r'"(.*?)"',
        ]

        for pattern in title_patterns:
            match = re.search(pattern, entry)
            if match:
                title = match.group(1).strip()
                # Remove any remaining LaTeX commands
                title = re.sub(r'\\[a-zA-Z]+(\[.*?\])?({.*?})?', '', title)
                break

        if title:
            titles.append(title)
        else:
            titles.append("Title not found")

    return titles

def extract_reference_titles(arxiv_id):
    cited_titles = []
    tar_file = None

    try:
        tar_file = download_arxiv_source(arxiv_id)
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with tarfile.open(tar_file, 'r:gz') as tar:
                    tar.extractall(path=temp_dir)
            except tarfile.ReadError:
                # If it's not a gzip file, try opening it as a regular tar file
                with tarfile.open(tar_file, 'r:') as tar:
                    tar.extractall(path=temp_dir)

            # Find the main .tex file
            main_tex_file = find_main_tex_file(temp_dir)

            if not main_tex_file:
                print(f"Could not find main .tex file for {arxiv_id}")
                return cited_titles

            # Process the main .tex file and all its inputs
            with open(main_tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                tex_content = process_input_commands(f.read(), os.path.dirname(main_tex_file))

            ref_files = find_reference_files(tex_content)
            if not ref_files:
                print(f"No reference files found in the processed .tex content for {arxiv_id}")
                return cited_titles

            # Extract titles from each referenced file
            for ref_file in ref_files:
                ref_path = os.path.join(os.path.dirname(main_tex_file), ref_file)
                if not os.path.exists(ref_path):
                    for ext in ['.bib', '.bbl']:
                        test_path = ref_path + ext
                        if os.path.exists(test_path):
                            ref_path = test_path
                            break

                if os.path.exists(ref_path):
                    try:
                        with open(ref_path, 'r', encoding='utf-8', errors='ignore') as ref_file:
                            content = ref_file.read()
                            if ref_path.endswith('.bib'):
                                titles = parse_bib_content(content)
                            elif ref_path.endswith('.bbl'):
                                titles = parse_bbl_content(content)
                            else:
                                print(f"Unrecognized reference file type: {ref_path}")
                                continue
                            cited_titles.extend(titles)
                    except UnicodeDecodeError:
                        print(f"Could not read file {ref_path} as UTF-8")

    except Exception as e:
        print(f"An error occurred while processing {arxiv_id}: {str(e)}")
    finally:
        if tar_file and os.path.exists(tar_file):
            os.unlink(tar_file)

    return cited_titles

@backoff.on_exception(backoff.expo, RequestException, max_tries=3)
def get_arxiv_abstract(arxiv_id):
    try:
        base_url = "http://export.arxiv.org/api/query?"
        query = f"id_list={arxiv_id}"

        response = requests.get(base_url + query)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', namespace)

        if entry is None:
            print(f"No entry found for ArXiv ID: {arxiv_id}")
            return None

        summary = entry.find('atom:summary', namespace)
        if summary is not None:
            return summary.text.strip()
        else:
            print(f"No abstract found for ArXiv ID: {arxiv_id}")
            return None

    except Exception as e:
        print(f"Error retrieving abstract for ArXiv ID {arxiv_id}: {str(e)}")
        return None

def get_random_user_agent():
    """Return a random user agent string."""
    return random.choice(USER_AGENTS)

def get_headers():
    """Return headers that mimic a browser."""
    return {
        'User-Agent': get_random_user_agent(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

@backoff.on_exception(backoff.expo, RequestException, max_tries=3)
def find_arxiv_id_from_title(title):
    try:
        clean_title = re.sub(r'[^\w\s]', ' ', title)
        clean_title = re.sub(r'\s+', ' ', clean_title).strip()
        query = '+AND+'.join([f'ti:{word}' for word in clean_title.split()])

        url = f"http://export.arxiv.org/api/query?search_query={query}&max_results=5"

        response = requests.get(url, headers=get_headers())
        response.raise_for_status()
        root = ET.fromstring(response.content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}

        best_score = 0
        best_arxiv_id = None

        for entry in root.findall('.//atom:entry', namespace):
            entry_title = entry.find('atom:title', namespace).text.strip()
            similarity = calculate_title_similarity(title.lower(), entry_title.lower())

            if similarity > best_score:
                best_score = similarity
                id_url = entry.find('atom:id', namespace).text
                best_arxiv_id = id_url.split('/')[-1]

            if similarity > 90:
                break

        if best_score > 70:
            return best_arxiv_id
        else:
            return None

    except Exception as e:
        print(f"Error finding ArXiv ID: {str(e)}")
        return None

def calculate_title_similarity(title1, title2):
    """Calculate a simple similarity score between two titles."""
    t1 = re.sub(r'[^\w\s]', '', title1.lower())
    t2 = re.sub(r'[^\w\s]', '', title2.lower())

    words1 = set(t1.split())
    words2 = set(t2.split())

    if not words1 or not words2:
        return 0

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    return (intersection / union) * 100

@backoff.on_exception(backoff.expo, RequestException, max_tries=3)
def get_abstract_from_arxiv(arxiv_id):
    try:
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        response = requests.get(url, headers=get_headers())
        response.raise_for_status()

        root = ET.fromstring(response.content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        summary = root.find('.//atom:summary', namespace)

        if summary is not None and summary.text:
            # Clean up the abstract text
            abstract = summary.text.strip()
            abstract = re.sub(r'\s+', ' ', abstract)
            return abstract

        return None

    except Exception as e:
        print(f"Error retrieving abstract from ArXiv API: {str(e)}")
        return None

@backoff.on_exception(backoff.expo, RequestException, max_tries=3)
def get_abstract_from_arxiv_webpage(arxiv_id):
    try:
        url = f"https://arxiv.org/abs/{arxiv_id}"
        response = requests.get(url, headers=get_headers())
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        abstract_element = soup.select_one('blockquote.abstract')

        if abstract_element:
            for descriptor in abstract_element.select('span.descriptor'):
                descriptor.extract()

            abstract_text = abstract_element.get_text(strip=True)
            abstract_text = re.sub(r'\s+', ' ', abstract_text)
            return abstract_text

        return None

    except Exception as e:
        print(f"Error scraping abstract from ArXiv page: {str(e)}")
        return None

@backoff.on_exception(backoff.expo, RequestException, max_tries=3)
def get_google_scholar_info(title):
    search_query = scholarly.search_pubs(title)
    first_result = next(search_query, None)
    
    if first_result:
        # Get detailed information
        paper_details = scholarly.fill(first_result)
        
        result_title = paper_details.get('bib', {}).get('title', "Title not found")
        authors = paper_details.get('bib', {}).get('author', ["Authors not found"])
        authors_info = "; ".join(authors) if isinstance(authors, list) else authors
        abstract = paper_details.get('bib', {}).get('abstract', "Abstract not found")
        result_url = paper_details.get('pub_url', None)
        
        similarity = calculate_title_similarity(title, result_title)
        if similarity < 60:
            print(f"Warning: Found paper may not match query. Similarity score: {similarity}%")
                
        return {
            'title': result_title,
            'authors': authors_info,
            'abstract': abstract,
            'url': result_url,
            'similarity_score': similarity
        }
    
    # Web scraping if scholarly fails
    try:
        query = urllib.parse.quote_plus(title)
        url = f"https://scholar.google.com/scholar?q={query}&hl=en"

        response = requests.get(url, headers=get_headers())
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='gs_ri')

        if not results:
            return None

        first_result = results[0]

        # Extract the paper's title
        result_title_element = first_result.find('h3', class_='gs_rt')
        result_title = result_title_element.get_text(strip=True) if result_title_element else "Title not found"
        result_title = re.sub(r'^\[[A-Z]+\]\s*', '', result_title)

        # Extract the URL
        result_url_element = result_title_element.find('a') if result_title_element else None
        result_url = result_url_element['href'] if result_url_element and 'href' in result_url_element.attrs else None

        # Extract authors and publication info
        authors_element = first_result.find('div', class_='gs_a')
        authors_info = authors_element.get_text(strip=True) if authors_element else "Authors not found"

        # Extract the abstract
        abstract_element = first_result.find('div', class_='gs_rs')
        abstract = abstract_element.get_text(strip=True) if abstract_element else "Abstract not found"

        similarity = calculate_title_similarity(title, result_title)
        if similarity < 60:
            print(f"Warning: Found paper may not match query. Similarity score: {similarity}%")

        return {
            'title': result_title,
            'authors': authors_info,
            'abstract': abstract,
            'url': result_url,
            'similarity_score': similarity
        }

    except Exception as e:
        print(f"Error retrieving information from Google Scholar: {str(e)}")
        return None
    finally:
        time.sleep(random.uniform(1, 3))

@backoff.on_exception(backoff.expo, RequestException, max_tries=2)
def extract_abstract_from_site(url):
    if not url or url.lower().endswith('.pdf'):
        return None

    try:
        response = requests.get(url, headers=get_headers(), timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        abstract_selectors = [
            'div.abstract', 'section.abstract', 'div#abstract', 'section#abstract',
            'div[class*="abstract"]', 'section[class*="abstract"]', 'p.abstract',
            'div.summary', 'div#summary', '.abstract-content', '#abstractSection',
            '.abstractSection', '.abstract-section', 'meta[name="description"]',
            'meta[property="og:description"]', '.abstract-text', '#abstract-content'
        ]

        for selector in abstract_selectors:
            elements = soup.select(selector)
            if elements:
                if selector.startswith('meta'):
                    return elements[0].get('content', '')
                return elements[0].get_text(strip=True)

        return None

    except Exception as e:
        print(f"Error extracting abstract from URL: {str(e)}")
        return None

def get_full_paper_abstract(title):
    result = {
        'title': title,
        'abstract': None,
        'authors': None,
        'url': None,
        'abstract_source': None
    }

    try:
        arxiv_id = find_arxiv_id_from_title(title)
        if arxiv_id:
            abstract = get_abstract_from_arxiv(arxiv_id)
            if not abstract:
                abstract = get_abstract_from_arxiv_webpage(arxiv_id)

            if abstract:
                result['abstract'] = abstract
                result['abstract_source'] = 'arxiv'
                result['url'] = f"https://arxiv.org/abs/{arxiv_id}"
                result['arxiv_id'] = arxiv_id
                return result

        gs_info = get_google_scholar_info(title)

        if gs_info:
            result.update(gs_info)
            result['abstract_source'] = 'google_scholar'

            gs_abstract = gs_info['abstract']
            if not gs_abstract or gs_abstract == "Abstract not found" or '…' in gs_abstract or len(gs_abstract) < 200:
                # Get full abstract from the paper URL
                if gs_info['url']:
                    site_abstract = extract_abstract_from_site(gs_info['url'])
                    if site_abstract and len(site_abstract) > len(gs_abstract):
                        result['abstract'] = site_abstract
                        result['abstract_source'] = 'publisher_site'

        return result

    except Exception as e:
        print(f"Error retrieving full paper abstract: {str(e)}")
        return result
    finally:
        # Add a delay to avoid being blocked
        time.sleep(random.uniform(1, 2))

def build_tree(arxiv_id, title, k=1):
    setup_proxy()

    root_paper_info = get_full_paper_abstract(title)
    root_abstract = root_paper_info['abstract'] if root_paper_info['abstract'] else "N/A"
    
    node_features = []  # Each node has a feature vector of shape (1, 768), BERT
    edge_index = [[], []] 
    node_arxiv_ids = []
    node_titles = []
    node_abstracts = []

    node_features.append(torch.ones(1, 768))
    node_arxiv_ids.append(arxiv_id)
    node_titles.append(title)
    node_abstracts.append(root_abstract)
    
    current_level_start = 0
    current_level_end = 1  

    for level in range(k):
        next_level_start = current_level_end
        
        # Process each node in the current level
        for node_idx in range(current_level_start, current_level_end):
            current_arxiv_id = node_arxiv_ids[node_idx]
            if current_arxiv_id == "N/A":
                continue
            
            # Extract reference titles for the current node
            try:
                reference_titles = extract_reference_titles(current_arxiv_id)
                
                for ref_title in reference_titles:
                    ref_arxiv_id = get_arxiv_id(ref_title) or "N/A"
                    ref_paper_info = get_full_paper_abstract(ref_title)
                    ref_abstract = ref_paper_info['abstract'] if ref_paper_info['abstract'] else "N/A"
                    
                    if ref_arxiv_id == "N/A" and 'arxiv_id' in ref_paper_info and ref_paper_info['arxiv_id']:
                        ref_arxiv_id = ref_paper_info['arxiv_id']
                    
                    node_features.append(torch.ones(1, 768))
                    node_arxiv_ids.append(ref_arxiv_id)
                    node_titles.append(ref_title)
                    node_abstracts.append(ref_abstract)
                    
                    target_idx = len(node_features) - 1
                    edge_index[0].append(node_idx) 
                    edge_index[1].append(target_idx) 
            
            except Exception as e:
                print(f"Error processing node {node_idx} (ArXiv ID: {current_arxiv_id}): {str(e)}")
        
        current_level_start = current_level_end
        current_level_end = len(node_features)
        
        if current_level_start == current_level_end:
            break
    
    x = torch.cat(node_features, dim=0)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    graph = Data(
        x=x,
        edge_index=edge_index,
        arxiv_id=node_arxiv_ids,
        title=node_titles,
        abstract=node_abstracts
    )
    
    return graph

def merge_graphs(graphs):
    title_to_idx = {}
    
    merged_node_features = []
    merged_edge_index = [[], []]
    merged_arxiv_ids = []
    merged_titles = []
    merged_abstracts = []
    
    # Process each graph
    for graph in graphs:
        titles = graph.title
        arxiv_ids = graph.arxiv_id
        abstracts = graph.abstract
        edges = graph.edge_index.tolist()
        features = graph.x
        
        current_to_merged_idx = {}
        
        for i, title in enumerate(titles):
            clean_title_str = clean_title(title)
            if clean_title_str in title_to_idx:
                merged_idx = title_to_idx[clean_title_str]
                current_to_merged_idx[i] = merged_idx
                
                if arxiv_ids[i] != "N/A" and merged_arxiv_ids[merged_idx] == "N/A":
                    merged_arxiv_ids[merged_idx] = arxiv_ids[i]
                
                if abstracts[i] != "N/A" and merged_abstracts[merged_idx] == "N/A":
                    merged_abstracts[merged_idx] = abstracts[i]
            else:
                merged_idx = len(merged_node_features)
                merged_node_features.append(features[i].unsqueeze(0))
                merged_arxiv_ids.append(arxiv_ids[i])
                merged_titles.append(title)
                merged_abstracts.append(abstracts[i])
                
                title_to_idx[clean_title_str] = merged_idx
                current_to_merged_idx[i] = merged_idx
        
        for j in range(len(edges[0])):
            src_idx = edges[0][j]
            dst_idx = edges[1][j]
            
            merged_src_idx = current_to_merged_idx[src_idx]
            merged_dst_idx = current_to_merged_idx[dst_idx]
            
            merged_edge_index[0].append(merged_src_idx)
            merged_edge_index[1].append(merged_dst_idx)
    
    x = torch.cat(merged_node_features, dim=0)
    edge_index = torch.tensor(merged_edge_index, dtype=torch.long)
    
    merged_graph = Data(
        x=x,
        edge_index=edge_index,
        arxiv_id=merged_arxiv_ids,
        title=merged_titles,
        abstract=merged_abstracts
    )
    
    return merged_graph