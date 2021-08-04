from pathlib import Path
import requests


# Code to download files from gdrive (source: https://stackoverflow.com/a/39225272)
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = {'id': id}, stream = True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


# Use previous code to download models
def download_models(target_folder='~/.u2net'):
    folder = Path(target_folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)  # If target folder does not exist, create it

    # check if models exist and download them if they don't
    model1 = folder / 'u2net.pth'
    if not model1.is_file():
        print("Model not found. Downloading model u2net.pth...")
        download_file_from_google_drive('1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ', model1)

    model2 = folder / 'u2netp.pth'
    if not model2.is_file():
        print("Model not found. Downloading model u2netp.pth...")
        download_file_from_google_drive('1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy', model2)

    model3 = folder / 'u2net_human_seg.pth'
    if not model3.is_file():
        print("Model not found. Downloading model u2net_human_seg.pth...")
        download_file_from_google_drive('1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P', model2)
