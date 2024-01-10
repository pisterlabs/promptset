import sys
import openai

def main(modified, added, deleted, renamed):
    print("Modified files:", modified)
    print("Added files:", added)
    print("Deleted files:", deleted)
    print("Renamed files:", renamed)
    print("Success!")

if __name__ == "__main__":
    modified = sys.argv[1].split()
    added = sys.argv[2].split()
    deleted = sys.argv[3].split()
    renamed = sys.argv[4].split()
    main(modified, added, deleted, renamed)