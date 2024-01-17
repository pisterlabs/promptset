import json, os, requests

libraries = ["cohere", "guidance", "anthropic", "llamaindex", "langchain", "openai"]

print(
"""
Result Counts from Github (Collected Manually)
    guidance: 1.9k
    anthropic: 1.8k
    llamaindex: 117
    cohere: 5.8k
    openai: 97.8k
    langchain: 64.5k
"""
)

all_lib_hrefs = set()
for lib in libraries:
    with open(f'data/scraping-2.0/results_{lib}.json') as f:
        data = json.load(f)
    
    # Delete the "~remaining_combinations~" key
    del data["~remaining_combinations~"]

    total_num_results = 0
    total_hrefs = set()

    for charCombo, results in data.items():
        num_result = results['num_results']
        hrefs = results['hrefs']

        # Convert num_result to int
        total_num_results += num_result

        # Count hrefs
        total_hrefs.update(hrefs)

        # print("MISMATCH!!!", charCombo, num_result, len(hrefs)) if num_result != len(hrefs) and len(hrefs) < 100 else None
        # print(f'Character Combo: {charCombo}; {num_result}') if num_result > 100 else None
    all_lib_hrefs.update(total_hrefs)

    print(f'Library: {lib}')
    print('\tTotal number of results:', total_num_results)
    print('\tTotal number of hrefs:', len(total_hrefs))

# STARTING DOWNLOAD ###########################################################
print('\nTotal number of hrefs:', len(all_lib_hrefs))

all_rawFileURLs = [href.replace("blob/", "").replace("https://github.com", "https://raw.githubusercontent.com") for href in all_lib_hrefs]
for i in range(len(all_rawFileURLs)):
    if "#" in all_rawFileURLs[i]:
        all_rawFileURLs[i] = "#".join(all_rawFileURLs[i].split("#")[:-1])

root_dir = "repos"
if not os.path.exists(root_dir):
    os.mkdir(root_dir)

count = 0
for url in all_rawFileURLs:
    url_split = url.split("/")

    # Getting repo name
    repo_name = "~".join(url_split[3:5])

    # Get filename addr for local storage
    filename_addr = url_split[6:]
    filename_addr = "~".join(filename_addr)

    # repo path
    repo_path = os.path.join(root_dir, repo_name)
    if not os.path.exists(repo_path):
        os.mkdir(os.path.join(root_dir, repo_name))

    # file path
    file_path = os.path.join(repo_path, filename_addr)
    if len(file_path) > 255:
        filename_addr = filename_addr.split("~")[-1]
        file_path = os.path.join(repo_path, filename_addr)

    # print("Checking")
    if not os.path.exists(file_path):
        try:
            r = requests.get(url, timeout=1)
            # Exception thrown before file is created. 
            # So, if file exists, it's safe to assume that it's been downloaded successfully.
            if r.status_code == 200:
                with open(file_path, "w") as f:
                    f.write(r.text)
            else:
                print("Error: ", r.status_code, repo_path, filename_addr)
        except Exception as e:
            print(e)
            print("Error: ", repo_path, filename_addr)
    # print("Done")

    if count % 100 == 0:
        print(count, end=" ")
    count += 1

print("Done")

# Checking Number of Files Downloaded #########################################
root_dir = "repos"
count = 0
for root, dirs, files in os.walk(root_dir):
    count += len(files)
print("Total number of files DOWNLOADED:", count)