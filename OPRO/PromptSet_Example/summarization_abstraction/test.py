import json

if __name__ == "__main__":
    data = "0/synthetic_dataset.json"
    with open(data, "r") as f:
        data = json.load(f)
    print(len(data))