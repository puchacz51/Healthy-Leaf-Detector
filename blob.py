import requests
import tqdm

def download(url: str, filename: str):
    with open(filename, 'wb') as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))

            # tqdm has many interesting parameters. Feel free to experiment!
            tqdm_params = {
                'desc': url,
                'total': total,
                'miniters': 1,
                'unit': 'B',
                'unit_scale': True,
                'unit_divisor': 1024,
            }
            with tqdm.tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    f.write(chunk)

download(
            'https://datasetkaggle.blob.core.windows.net/dataset/original.zip?sp=r&st=2024-06-16T08:50:41Z&se=2024-06-16T16:50:41Z&spr=https&sv=2022-11-02&sr=b&sig=5u5GAPNcpkKS5wlRup8RdZOZ6t8ItO%2BciyHWJF4urdQ%3D'

         , '100MB.bin')