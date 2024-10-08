import json
import requests
import os
import shutil
from googlesearch import search
from bs4 import BeautifulSoup


def get_page_content(url):
    try:
        response = requests.get(url)
        if response:
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            # Get text
            text = soup.get_text()
            # Break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)
            return cleaned_text
        else:
            raise Exception("No response from the server.")
    except Exception as e:
        print(f"Error while fetching and cleaning content from {url}: {e}")
        return None


def empty_tmp_directory():
    try:
        folder = '/tmp'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        print("Temporary directory emptied.")
    except Exception as e:
        print(f"Error while emptying /tmp directory: {e}")


def save_content_to_tmp(content, filename):
    try:
        if content is not None:
            with open(f'/tmp/{filename}', 'w', encoding='utf-8') as file:
                file.write(content)
            print(f"Saved {filename} to /tmp")
            return f"Saved {filename} to /tmp"
        else:
            raise Exception("No content to save.")
    except Exception as e:
        print(f"Error while saving {filename} to /tmp: {e}")


def search_google(query):
    try:
        search_results = []
        for j in search(query, sleep_interval=5, num_results=10):
            search_results.append(j)
        return search_results
    except Exception as e:
        print(f"Error during Google search: {e}")
        return []


def handle_search(event):
    input_text = event.get('inputText', '')  # Extract 'inputText'

    # Empty the /tmp directory before saving new files
    print("Emptying temporary directory...")
    empty_tmp_directory()

    # Proceed with Google search
    print("Performing Google search...")
    urls_to_scrape = search_google(input_text)

    aggregated_content = ""
    results = []
    for url in urls_to_scrape:
        print("URLs Used: ", url)
        content = get_page_content(url)
        if content:
            print("CONTENT: ", content)
            filename = url.split('//')[-1].replace('/', '_') + '.txt'  # Simple filename from URL
            aggregated_content += f"URL: {url}\n\n{content}\n\n{'='*100}\n\n"
            results.append({'url': url, 'status': 'Content aggregated'})
        else:
            results.append({'url': url, 'error': 'Failed to fetch content'})

    # Define a single filename for the aggregated content
    aggregated_filename = f"aggregated_{input_text.replace(' ', '_')}.txt"
    # Save the aggregated content to /tmp
    print("Saving aggregated content to /tmp...")
    save_result = save_content_to_tmp(aggregated_content, aggregated_filename)
    if save_result:
        results.append({'aggregated_file': aggregated_filename, 'tmp_save_result': save_result})
    else:
        results.append(
            {
                'aggregated_file': aggregated_filename,
                'error': 'Failed to save aggregated content to /tmp',
            }
        )

    return {"results": results}


def handler(event, context):
    print("THE EVENT: ", event)

    response_code = 200
    if event.get('apiPath') == '/search':
        result = handle_search(event)
    else:
        response_code = 404
        result = {"error": "Unrecognized api path"}

    response_body = {'application/json': {'body': json.dumps(result)}}

    action_response = {
        'actionGroup': event['actionGroup'],
        'apiPath': event['apiPath'],
        'httpMethod': event['httpMethod'],
        'httpStatusCode': response_code,
        'responseBody': response_body,
    }

    api_response = {'messageVersion': '1.0', 'response': action_response}
    print("RESPONSE: ", action_response)

    return api_response
