import re
import json
import sys
import srt
import os
import openai
import time
import tiktoken
import threading
import queue
import concurrent.futures
from dotenv import load_dotenv
from ids import getAllIdsWowhead
from wowhead import getData
from bs4 import BeautifulSoup, NavigableString, Tag

# Constants
# Thread-safe queue for subtitles
fetch_queue = queue.Queue()



faction_description_regex = re.compile(r"\"(.*)\",\n\s*\"article-all\"")
faction_backup_description_regex = re.compile(r"<meta name=\"description\" content=\"(.*?)\">")
faction_g_faction_regex = re.compile(r"g_factions\[\d*\], (.*)\);")
#? This all works, but only English has any information...
                  # ADD AFTER THIS LINE ->
                  # description = None
                  # try:
                  #   # Get description
                  #   description = faction_description_regex.search(rawData).group(1)
                  #   # Replace all HTML tags
                  #   description = re.sub(r"\[.*?\]", "", description)
                  #   # Replace \\r\\n with \n
                  #   description = description.replace("\\r\\n", "\n")
                  #   # Replace many \n with just one
                  #   description = re.sub(r"\n+", "\n", description)
                  #   # Remove trailing newline
                  #   description = description.strip()
                  #   # Add description to g_faction
                  #   g_faction["description"] = description
                  # except Exception as e1:
                  #   try:
                  #     description = faction_backup_description_regex.search(rawData).group(1)
                  #     # Add description to g_faction
                  #     g_faction["description"] = description
                  #   except Exception as e2:
                  #     print(f"Failed to get description for {idType} {id} Exception: {e1} and {e2}")

# Creating a lookup table for "Description", "Progress", and "Completion"

skip_lookup_table = {
  "Guides",
  "Guias",
  "Руководства",
  "Guides",
  "가이드",
  "Guías",
  "指南",
}

lookup_table = {
    # English
    "Description": "Description",
    "Progress": "Progress",
    "Completion": "Completion",
    "Rewards": "Rewards",

    # Portuguese
    "Descrição": "Description",
    "Progresso": "Progress",
    "Completo": "Completion",
    "Recompensas": "Rewards",
    "Ganancias": "Rewards",

    # Russian
    "Описание": "Description",
    "Прогресс": "Progress",
    "Завершено": "Completion",
    "Награды": "Rewards",
    "Дополнительные награды": "Rewards",

    # German
    "Beschreibung": "Description",
    "Fortschritt": "Progress",
    "Vervollständigung": "Completion",
    "Belohnungen": "Rewards",

    # Korean
    "서술": "Description",
    "진행 상황": "Progress",  # "진행 상황" and "보상" both map to "Progress"
    "보상": "Progress",
    "완료": "Completion",  # "완료" and "획득" both map to "Completion"
    "획득": "Completion",
    # Rewards: "보상" is already included in the table, mapping to "Progress".
    # This demonstrates a case where the same word can have multiple meanings based on context.

    # Spanish
    "Descripción": "Description",
    "Progreso": "Progress",
    "Terminación": "Completion",
    "Recompensas": "Rewards",  # Same as in Portuguese
    "Ganancias": "Rewards",  # Same as in Portuguese

    # Chinese
    "描述": "Description",
    "奖励": "Progress",
    "进度": "Progress",
    "收获": "Completion",
    "达成": "Completion",
    # Rewards: "奖励" is already in the table, mapping to "Progress".
    # Similar to Korean, this word has multiple meanings.

    # French
    "Description": "Description",  # Same in English
    "Progrès": "Progress",
    "Achèvement": "Completion",
    "Récompenses": "Rewards",
    "Gains": "Rewards",  # Same as in Spanish
}

def getQuestSections(locale, data, id, idType="quest"):
  # Use BeautifulSoup to parse the HTML content
  soup = BeautifulSoup(data, 'lxml')

  # Remove all script tags
  for script in soup.find_all("script"):
      script.decompose()

  # Find the div with class "text"
  text_div = soup.find('div', class_='text')

  sections = {}
  # Check if the div was found
  if text_div:
      # Extract the title from the first <h1 class="heading-size-1"> element
      h1_tag = text_div.find('h1', class_='heading-size-1')
      if h1_tag:
          title = h1_tag.get_text(strip=True)
          # print(f"Title: {title}")
          sections["Title"] = title

          # Extract the text following the <h1> tag until the next non-<a> element
          quest_text = []
          for element in h1_tag.next_siblings:
              if isinstance(element, Tag):
                  if element.name != 'a':
                      break
                  quest_text.append(element.get_text())
              elif isinstance(element, NavigableString):
                  text = element.strip()
                  if text:
                      quest_text.append(element)

          # print("Quest Text:", ''.join(quest_text))
          sections["Text"] = ''.join(quest_text)
      current_h2_text = None
      current_content = []

      # Iterate over all elements in the div
      for element in text_div.children:
          if "Rewards" in sections:
            break
          if isinstance(element, Tag):
              if element.name == 'h2' and 'heading-size-3' in element.get('class', []):
                  # Save previous section if it exists
                  if current_h2_text is not None:
                      # Break if we have 3 sections
                      if len(sections) == 3:
                        break
                      section_title = lookup_table.get(current_h2_text)
                      if section_title is None:
                        print(f"Section title is None for ({locale}:'{current_h2_text}') - {idType} {id}")
                      else:
                        sections[section_title] = ' '.join(current_content)
                      current_content = []

                  # Update current heading text
                  current_h2_text = element.get_text(strip=True)
              elif current_h2_text is not None:
                  current_content.append(element.get_text())
          elif isinstance(element, NavigableString) and current_h2_text is not None:
              text = element.strip()
              if text:
                  current_content.append(element)

      # Add the last section
      if current_h2_text is not None and len(sections) < 3 and current_h2_text not in skip_lookup_table:
          section_title = lookup_table.get(current_h2_text)
          if section_title is None:
            print(f"Section title is None for ({locale}:'{current_h2_text}') - {idType} {id}")
          else:
            sections[section_title] = ' '.join(current_content)

      for section_title, section_content in sections.items():
          # Remove double spaces
          section_content = re.sub(r"\s+", " ", section_content)
          # print(f"Section: {section_title}\nContent:\n{section_content}\n")
          # print(section_title)
          # continue
  else:
      print(f"No 'div' with class 'text' found. {idType} {id}")

  return sections

def fetch_worker(version, idData):
    while not fetch_queue.empty():
        idType, id = fetch_queue.get()
        try:
            if len(fetch_queue.queue) % 100 == 0:
                print(f"{len(fetch_queue.queue)} items left in queue")

            # Get data
            start_time = time.time()
            data = getData(idType, id, version, "all")

            # If data is None, continue to the next item in the queue
            if data is None:
                print(f"Data is None for {idType} {id}")
                continue

            # Common processing for all idTypes
            idData[idType][id] = {}
            if idType == "faction":
              for locale, data in data.items():
                  if type(data) == bytes:
                    rawData = data.decode("utf-8")
                  else:
                    rawData = data
                  # Get g_faction
                  g_faction = faction_g_faction_regex.search(rawData).group(1)
                  # Load g_faction as JSON
                  g_faction = json.loads(g_faction)

                  idData[idType][id][locale] = g_faction
            elif idType == "quest":
              usData = getQuestSections("enUS", data["enUS"], id)
              if len(usData) == 0:
                print(f"Section count is 0 for {idType} {id}")
                continue
              else:
                idData[idType][id]["enUS"] = usData
              for locale, localeData in data.items():
                if locale != "enUS":
                  localeData = getQuestSections(locale, localeData, id)
                  if len(localeData) == 0:
                    print(f"Section count is 0 for {idType} {id} {locale}")
                    continue
                  elif len(localeData) != len(usData):
                    print(f"Section count mismatch for {idType} {id} {locale}")
                  idData[idType][id][locale] = localeData
            else:
              for locale, localeData in data.items():
                  data = json.loads(localeData)
                  idData[idType][id][locale] = data
            print(f"{str(idType).capitalize()} {id} took {(time.time() - start_time):.2f} seconds")

        except Exception as e:
            print(f"Exception: {e} for {idType} {id}, requeueing...")
            fetch_queue.put((idType, id))
        finally:
            fetch_queue.task_done()


if __name__ == "__main__":
  # Classic, TBC, Wotlk
  version = "Classic"
  all_ids = {}
  all_ids["npc"] = getAllIdsWowhead(version, "npc")
  all_ids["item"] = getAllIdsWowhead(version, "item")
  all_ids["quest"] = getAllIdsWowhead(version, "quest")
  all_ids["object"] = getAllIdsWowhead(version, "object")
  all_ids["spell"] = getAllIdsWowhead(version, "spell")
  all_ids["faction"] = getAllIdsWowhead(version, "faction")

  # Save all ids
  with open(f"{version.lower()}_all_ids.json", "w", encoding="utf-8") as f:
    json.dump(all_ids, f, indent=2, ensure_ascii=False)

  for idType, ids in all_ids.items():
    # if idType != "quest":
      print(f"{idType}: {len(ids)}")
      for id in ids:
        fetch_queue.put((idType, id))
    # else:
      # print(f"{str(idType).capitalize()} is skipped for now, requires special handling")

  idData = {}
  for idType, ids in all_ids.items():
    idData[idType] = {}

  # Start translation workers
  num_threads = 10  # You can adjust this based on your actual RPM and CPU cores
  threads = []
  for _ in range(num_threads):
      thread = threading.Thread(target=fetch_worker, args=(version, idData))
      thread.start()
      threads.append(thread)
      # Stagger the start of the threads
      time.sleep(0.3)

  # Wait for all threads to finish
  for thread in threads:
      thread.join()

  # This function is used to write a dictionary to a file
  # But also not print the trailing comma
  # json.dump works but the map value is too nested and makes the file unreadable
  def write_dict(d, f, indent=0):
    f.write("{")
    items = list(d.items())
    for i, (k, v) in enumerate(items):
        f.write(f"\n{' ' * (indent + 2)}\"{k}\": ")
        if isinstance(v, dict):
            write_dict(v, f, indent=indent+2)
        else:
            f.write(json.dumps(v, ensure_ascii=False))
        # Write a comma if this isn't the last item
        if i < len(items) - 1:
            f.write(",")
    # If the dictionary is empty, don't print a newline
    if len(items) == 0:
      f.write("}")
    else:
      f.write(f"\n{' ' * indent}}}")

  # Why i don't just use json.dump() is because the map is too nested and creates a huge file
  filename = f"{version.lower()}_locales.json"
  print(f"Saving {filename}...")

  with open(filename, "w", encoding="utf-8") as f:
    write_dict(idData, f)

  print("Done")