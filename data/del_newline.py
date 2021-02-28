import csv

in_fnam = "Binance_BTCBUSD_1m_1609459200000-1614470400000.csv"
out = "nBinance_BTCBUSD_1m_1609459200000-1614470400000.csv"
with open(in_fnam) as in_file:
    with open(out, 'w') as out_file:
        writer = csv.writer(out_file)
        for row in csv.reader(in_file):
            if any(row):
                writer.writerow(row)