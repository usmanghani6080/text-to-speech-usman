# text-to-speech (main)

Text-to-speech module's implementation on AWS or other servers. 

#### Cloning the Main Repository :
```
git clone -b main https://github.com/usmanghani6080/text-to-speech-usman.git
```

#### Navigate to Main Repository :
```
cd text-to-speech-usman
```

#### Installing python :
```
sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.10 -y
sudo apt install python3.10-venv -y
sudo apt install python3.10-dev -y
```
#### Creating virtual Enviroment :
```
python3.10 -m venv tts-venv
source tts-venv/bin/activate
```
#### Installing requirements and dependencies:
```
pip install -r requirements.txt
```

#### Running API Code:
```
uvicorn app:app --host 0.0.0.0 --port <any> --reload
```