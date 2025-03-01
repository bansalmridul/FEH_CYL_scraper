import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import numpy as np

def scrape_poll_results(url):

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        results = defaultdict(int)
        soup = BeautifulSoup(response.content, "html.parser")  # Parse the HTML

        # *** KEY: Identify the elements containing the poll data ***
        # This is the most critical and site-specific part.  Use your browser's
        # developer tools (right-click, "Inspect" or "Inspect Element") to
        # examine the website's HTML structure and find the elements that hold
        # the poll results (e.g., question text, candidate/option names, vote counts, percentages).

        # Example: Assuming results are in a table with class "poll-results"
        results_table = soup.find("table", class_="result-ranking-table")
        if results_table:
            for row in results_table.find_all("tr"):  # Iterate through table rows
                cells = row.find_all("td")  # Find all cells within each row
                if cells:  # Check if the row has any cells (to avoid empty rows)
                    # Extract data from the cells (adapt as needed)
                    # Example:
                    rank = cells[0].text.strip()
                    name = cells[1].text.strip()
                    votes = int(cells[2].text.strip().replace(",", "").strip())

                    titles = []
                    if "Heroes" in name:
                        titles.append("Heroes")
                    if "Tokyo" in name:
                        titles.append("TMS")
                    if "Shadow Dragon" in name or "Mystery" in name:
                        titles.append("Archanea")
                    if "Valentia" in name:
                        titles.append("SoV")
                    if "Holy War" in name:
                        titles.append("Genealogy")
                    if "Thracia" in name:
                        titles.append("Thracia")
                    if "Binding" in name:
                        titles.append("Binding")
                    if "Blazing" in name:
                        titles.append("Blazing")
                    if "Sacred" in name:
                        titles.append("Sacred")
                    if "Radian" in name:
                        titles.append("Tellius")
                    if "Awakening" in name:
                        titles.append("Awakening")
                    if "Fates" in name:
                        titles.append("Fates")
                    if "Three Houses" in name:
                        titles.append("3Houses")
                    if "Hopes" in name and "Houses" not in name:
                        titles.append("3Hopes")
                    if "Engage" in name:
                        titles.append("Engage")
                    
                    entries = len(titles)
                    if entries == 0:
                        print("ERROR")
                        print(rank, name, votes)
                        raise ValueError("Missed a title")
                    for title in titles:
                        results[title] += votes // entries              
                else:
                    print("not in cell")
        else:
            print("Poll results table not found.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return results

def makeGraph(resultsTop200, results201_397):
    resultsAll = defaultdict(int)
    for key in resultsTop200.keys():
        resultsAll[key] = results201_397[key] + resultsTop200[key]
    votesTotal = sum(resultsAll.values())
    votes200 = sum(resultsTop200.values())
    print(resultsAll)
    print(resultsTop200)
    for data in [resultsAll, resultsTop200]:
        # Combine itemPart1 and itemPart2
        combined_value = data["3Houses"] + data["3Hopes"]
        data["Fodlan"] = combined_value
        del data["3Houses"]
        del data["3Hopes"]
    
    # Plot both dictionaries
    plot_dict(resultsAll, "CYL Votes By Title")
    plot_dict(resultsTop200, "CYL Votes By Title (Top 200)")

def plot_dict(data, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    pairs = [(value, key) for key, value in data.items()]
    pairs.sort()
    values = [pair[0] for pair in pairs]
    keys = [pair[1] for pair in pairs]
    bar_positions = np.arange(len(keys))
    
    for i, value in enumerate(values):
        ax.bar(i, value, width=0.6)
        ax.text(i, value, f"{value}", ha='center', va='bottom')

    # Set labels and title
    ax.set_title(title)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(keys, rotation=45, ha='right')
    ax.set_ylabel("Votes")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    poll_url = "https://vote9.campaigns.fire-emblem-heroes.com/en-US/result-detail/1"  # Replace with the actual URL
    poll_url2 = "https://vote9.campaigns.fire-emblem-heroes.com/en-US/result-detail/2"  # Replace with the actual URL
    resultsTop200 = scrape_poll_results(poll_url)
    time.sleep(2)
    results201_397 = scrape_poll_results(poll_url2)
    makeGraph(resultsTop200, results201_397)

    
