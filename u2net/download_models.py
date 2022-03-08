from pathlib import Path
import gdown


def download_models(target_folder='~/.u2net'):
    folder = Path(target_folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)  # If target folder does not exist, create it

    # check if models exist and download them if they don't
    model1 = folder / 'u2net.pth'
    if not model1.is_file():
        print("Model not found. Downloading model u2net.pth...")
        gdown.download(id='1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ', output=str(model1))

    model2 = folder / 'u2netp.pth'
    if not model2.is_file():
        print("Model not found. Downloading model u2netp.pth...")
        gdown.download(id='1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy', output=str(model2))

    model3 = folder / 'u2net_portrait.pth'
    if not model3.is_file():
        print("Model not found. Downloading model u2net_portrait.pth...")
        gdown.download(id='1IG3HdpcRiDoWNookbncQjeaPN28t90yW', output=str(model3))

