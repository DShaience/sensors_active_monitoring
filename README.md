# Sensors Active Monitoring v1.01
Sensor Active Monitoring (SAM) is designed to visualize and analyze the sensors logs from GPU-Z, and alert over potential issues before damage occurs.
This can be abnormally high temperature (GPU, CPU, RAM, etc.) and any other sensor.<br>
SAM is using Streamlit to provide a simple interactive Web-GUI, and allows for dynamic sensors' selection. <br> 
The data can be filtered by time, and using Streamlit's graphs it enables hover and zoom. 

# Screenshots

## The Main Page
![Main screen](screenshots/Main_screen.png)
<br><br>
## Selecting Sensors
![Main screen](screenshots/Sensor_selections.png)
<br><br>
## Selecting by Hours
![Main screen](screenshots/Hours_selection.png)
<br><br>
## GPU Prefetch Cap Reason
![Main screen](screenshots/Prefetch.png)

## Try it out!
You may upload your own GPU-Z logs into the analyzer [here](https://sensors-monitoring-webapp.azurewebsites.net/).<br>
We don't retain any data or files (in the future we may include some high-level logs, see "Data" section below). <br>
GPU-Z itself can be downloaded from [here](https://www.techpowerup.com/gpuz/)

## Data
In the future we may want to add some aggregate data logging. In any case none of the actual raw data is saved <br>
and the enable/disable option for aggregation-logs will be explicitly shown on the screen. <br>
We're really not here to steal anyone's data. Voluntarily shared data is used to come up with better and more <br>
useful analytics.


## Installation
* Clone the repo
* Create a conda environment using the requirements.txt file. Tested with Python >= 3.10.
``` 
pip install -r requirements.txt
```
* The main file is sensors_active_monitoring\src\webapp\app.py you may run it using commandline:

``` 
cd c:\myprojects\sensors_active_monitoring\src
c:\myprojects\my_conda_envs\sensor_active_monitoring_env\python.exe run.py
``` 
* This will open your browser with the main screen.
* Select a GPU-Z Log file and load it into Sensor Active Monitoring


## Future Release Plans
* Add more analytics and options
* Add a better time filtering, perhaps with range
* Add dates range above the charts (since Streamlit plots have difficulty with dates)
* Add logs (debugging only)

## Project
This project is developed and maintained by [Shay Amram](https://www.linkedin.com/in/shay-amram-69924051/)
