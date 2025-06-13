# AeroAI: AI-Powered Anomaly Detection for Air Traffic Control

**AeroAI is an advanced end-to-end software solution designed to enhance airspace safety and reduce cognitive load for air traffic controllers. By leveraging artificial intelligence and machine learning, AeroAI proactively identifies, classifies, and explains anomalous flight behaviors in real-time, particularly within congested airspace.**

## The Challenge: The Demands of Modern Air Traffic Control

The role of an Air Traffic Controller (ATC) is paramount to the safety and efficiency of global air travel. ATCs are tasked with managing a high volume of aircraft, ensuring safe separation, and providing precise instructions within dynamic and often congested airspace. This inherently demanding role leads to significant cognitive load and stress, especially during peak traffic or unexpected events. While traditional systems provide vast amounts of data, the critical task of identifying subtle deviations or predicting potential anomalies often rests heavily on the controller's vigilance and experience, which can be challenged under pressure. Failure to promptly recognize anomalous behavior—such as unexpected deviations from flight paths, incorrect altitudes, or unusual speed changes—can have critical safety implications.

## Our Solution: Intelligent Assistance with AeroAI
"AeroAI's architecture is designed for real-time processing and explainable anomaly detection:"

### Data Ingestion & Initial Filtering (Aerhose):
* "The system ingests real-time flight data (with a standard ~30s regulatory delay) via **FlightAware's Firehose data feed**."
* "The first component, **Aerhose**, establishes this connection and performs crucial initial filtering:"
    * "It isolates **positional updates**, as these are the core data AeroAI interprets for trajectory analysis."
    * "It applies **geographic filtering**, processing only those flights operating within a specified radius of a defined geographic coordinate, allowing for targeted airspace monitoring."

### Preprocessing & Anomaly Detection (AeroPredictor & AeroEncoder):
* "Filtered data is then passed to **AeroPredictor**. This component preprocesses the flight data, preparing a specific set of features to be fed into our anomaly detection engine. These features include:"
    * Aircraft Ground Speed
    * Heading
    * Altitude
    * Vertical Rate
    * Rate of Change of Ground Speed
    * Rate of Change of Heading
    * Status of Altitude Change
* "The core of our detection system is **AeroEncoder**, a custom-designed autoencoder neural network. Trained on normal flight patterns, AeroEncoder identifies anomalous behavior by detecting significant deviations in an aircraft's current feature set from these learned norms."

### Explainability & Output Generation:
* "When AeroEncoder flags a flight as anomalous, it also utilizes **saliency mapping**. This technique provides a description of *why* the behavior was considered anomalous by highlighting which input features most contributed to the detection."
* "AeroPredictor then compiles the anomaly status, the saliency map explanation, and other necessary flight data into a structured **JSON dictionary**."

### Data Storage & Real-Time Notification (Redis & Pub/Sub):
* "This JSON output is sent to a **Redis database**. For each flight, data is stored as a list of JSON serialized dictionaries, accessible via a key."
* "Crucially, upon each new insertion into Redis, the key for the new data is **published to a Redis channel named `lpush_channel`**. This efficient publish/subscribe mechanism instantly notifies any subscribed components of new flight updates."

### Visualization:
* "Subscribers to `lpush_channel`, such as our dedicated User Interface (UI), receive these real-time notifications. The UI then retrieves the detailed data from Redis to visualize the flights, their behavior, and clearly present any detected anomalies along with their explanations to the air traffic controller."

## Installation
This application only works on Linux and distributions of Linux (e.g. Ubuntu)
### Clone the repository and change directory.
```
git clone https://github.com/xKimChip/AeroAI.git
cd AeroAI
```

### (Optional) Create a virtual environment in Python 3.12
```
python3.12 -m venv <environment_name>
```
```
source ./<environment_name>/bin/activate
```
or
```
cd <environment_name>
source bin/activate
```
If you are using Conda, you can create a .yml file to auto configure
```
conda env create -f aero_ai.yml

conda activate .aeroai_env
```

### Install packages
If using Conda installation method, skip this
```
pip install -r requirements.txt
```
If you see `ERROR: Failed building wheel for pandas` try and then rerun above
```
sudo apt-get install build-essential python3-dev
```
For RHEL 8 instead use
```
sudo dnf groupinstall "Development Tools"
sudo dnf install python3-devel
```

then run
```
sudo apt update
sudo apt-get install python3-tk
sudo apt install redis-server
```
or if using RHEL 8
```
sudo dnf install python3-tkinter redis
```

## Running AeroAI
### Ensure Redis server is running
```
redis-server
```
RHEL 8 server start
```
sudo systemctl start redis
sudo systemctl enable redis
```
### Running Datafeed
Run Aerhose first in terminal
```
python3 Aerhose.py
```
### Running UI
Open a new terminal and run UI (make sure to activate environment if needed)
```
python3 app.py
```
Opening up `localhost:5001` on a browser should now should the fully running application







## Authors
| Team Member         | GitHub Username |
|---------------------|----------------|
| Bobak Sadri       | [bobaksadri](https://github.com/bobaksadri) |
| Royce Chipman      | [xKimchip](https://github.com/xKimchip) |
| Parsa Hesam        | [phesamuci](https://github.com/phesamuci) |
| Benjoseph Villamor | [bnvillamor](https://github.com/bnvillamor) |
| Junyoung Yoon      | [junbuggyTT](https://github.com/junbuggyTT) |
| Dushyant Bharadwaj | [dush1508](https://github.com/dush1508) |
| Alvin Phan         | [alvinatp](https://github.com/alvinatp) |
