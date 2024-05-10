# Run bash script to start the login node

# Importing the necessary libraries
import os
import subprocess
import time

# Running the bash script
start = time.time()
subprocess.run(["bash", "main.sh"])
end = time.time()
print(f"Time taken: {end-start}")

# End of file
