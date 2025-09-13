import pandas as pd
import csv

with open("training/Dec16-21-38-03/best/predictions_fix.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "TARGET"])
    with open("training/Dec16-21-38-03/best/predictions.csv", "r") as fIn:  
        heading = next(fIn) 
        reader = csv.reader(fIn)
        for row in reader: 
            id = row[0]
            results = row[1]
            final = []
            for res in results.split(','):
                if "-" in res:
                    # xx法xx-x條
                    pre, suf = res.split("-")
                    num = suf.split("條")[0]
                    final.append(f"{pre}條之{num}")
                else:
                    final.append(res)
            writer.writerow([id, ",".join(final)])
        
