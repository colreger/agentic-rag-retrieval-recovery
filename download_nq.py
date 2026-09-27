from beir import util

# name of the BEIR dataset we want to use
dataset = "nq"

# official BEIR download location	
url = ("https://public.ukp.informatik.tu-darmstadt.de/"
    f"thakur/BEIR/datasets/{dataset}.zip")

# download and unpack the dataset into a local folder called "datasets"
data_path = util.download_and_unzip(url, "datasets")

print("Dataset stored at:")
print(data_path)