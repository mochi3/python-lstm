import csv

def csvwriter(data):
    with open("usd_jpy_test.csv", "w") as csv_file:
        fieldnames = ["time", "volume", "o", "h", "l", "c"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
