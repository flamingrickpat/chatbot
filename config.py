import json

with open("config.json", "r") as jsonfile:
    data = json.load(jsonfile)
print(data)