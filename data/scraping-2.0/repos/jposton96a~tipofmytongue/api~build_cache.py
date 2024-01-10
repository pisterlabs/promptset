# import modules
import numpy as np
import openai
import os.path

out_cache_path = "res/word_embeddings_cache.npz"
word_dicts = "res/words.txt"

# open the file that contains the words
file = open(word_dicts, "r")
lines = file.readlines()
file.close()

def save_embeddings(file_path, embeddings):
  np_embeddings = embeddings

  # save the embeddings array to the .npz file using a keyword argument
  np.savez(file_path, embeddings=np_embeddings)
  del(np_embeddings)

def count_populated(a):
  for i, line in enumerate(a):
    if line.nonzero()[0].size == 0 or np.any(np.isnan(line)):
      return i

# define your input parameters
start = 0  # start line index
end = len(lines)  # end line index

# create an empty list to store the embeddings
embeddings = None
if os.path.isfile(out_cache_path):
  print("Detected existing checkpoint. Loading checkpoint...", end='')
  # Load the embeddings from the numpy array
  data = np.load(out_cache_path)
  embeddings = data["embeddings"]
  del(data)
  print("Done")

  checkpoint = count_populated(embeddings)
  print(f"Continuing from checkpoint index @ {checkpoint}")
  input("Press enter to proceed or ctrl+c to exit")
  start=checkpoint
else:
  print("Building cache from scratch...")
  embeddings = np.empty((len(lines), 1536))

# Free up some memory
lines_subset = lines[start:end]
del(lines)

# iterate over the sublist of lines based on the indices
for i, line in enumerate(lines_subset):
  # strip the newline character and assign it to text
  text = line.rstrip("\n")
  text_id = start + i
  response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
  embeddings[text_id] = response["data"][0]["embedding"]
  if i % 500 == 0 or i == len(lines_subset) - 1:
    print(f"Saving {i} (idx={text_id}) of {end-start} words... ", end='')
    save_embeddings(out_cache_path, embeddings)
    print("Done.")

# Final save
# save_embeddings(out_cache_path, embeddings)