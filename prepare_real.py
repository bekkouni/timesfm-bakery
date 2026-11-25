"""Make a small daily series from the original French bakery POS export."""
import argparse
import hashlib
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from generate_data import CONTEXT, DATA, HORIZON

SOURCE = "https://www.kaggle.com/datasets/matthieugimbert/french-bakery-daily-sales"
DOWNLOAD = "https://api.kaggle.com/v1/datasets/download/matthieugimbert/french-bakery-daily-sales"
CSV_HASH = "af5eede55b6eb2efb6bebb0e6dd1a51568c7eb73554d2f8ad2563081025681b2"
DAILY = DATA / "french_bakery_daily.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="?", help="Previously downloaded Kaggle zip")
    args = parser.parse_args()
    if args.archive:
        archive = args.archive
    else:
        with urlopen(DOWNLOAD, timeout=90) as response:
            archive = io.BytesIO(response.read())
    with zipfile.ZipFile(archive) as zipped:
        raw = zipped.read("Bakery sales.csv")
    if hashlib.sha256(raw).hexdigest() != CSV_HASH:
        raise ValueError("The source export has changed. Verify its contents before using it.")
    sales = pd.read_csv(io.BytesIO(raw), parse_dates=["date"])
    products = {"TRADITIONAL BAGUETTE": "loaf", "CROISSANT": "croissant", "PAIN AU CHOCOLAT": "chocolate"}
    products_sold = sales[sales.article.isin(products)]
    daily = products_sold.pivot_table(index="date", columns="article", values="Quantity", aggfunc="sum")
    tickets = sales.groupby("date").ticket_number.nunique()
    # A missing product on a recorded day sold zero. Entire missing dates are unknown.
    daily = daily.rename(columns=products).reindex(tickets.index).fillna(0)
    daily["tickets"] = tickets
    days = pd.date_range(end=sales.date.max(), periods=CONTEXT + HORIZON, name="date")
    data = daily.reindex(days)[["loaf", "croissant", "chocolate", "tickets"]].astype("Int64")
    DATA.mkdir(exist_ok=True)
    data.to_csv(DAILY, lineterminator="\n")
    print(f"{len(sales)} source rows; {data.loaf.iloc[-HORIZON:].count()}/{HORIZON} observed evaluation days.")


if __name__ == "__main__":
    main()
