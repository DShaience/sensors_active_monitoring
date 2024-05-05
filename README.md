# Sensors Active Monitoring v1.0
Sensor Active Monitoring (SAM) is designed to visualize and analyze the sensors logs from GPU-Z, and alert over potential issues before damage occurs.
This can be abnormally high temperature (GPU, CPU, RAM, etc.) and any other sensor.<br>
SAM is using Streamlit to provide a simple interactive Web-GUI, and allows for dynamic sensors' selection. <br> 
The data can be filtered by time, and using Streamlit's graphs it enables hover and zoom. 

## Screenshots

### The Main Page
![Main screen](screenshots/Main_screen.png)
<br><br>
### Selecting Sensors
![Main screen](screenshots/Sensor_selections.png)
<br><br>
### Selecting by Hours
![Main screen](screenshots/Hours_selection.png)
<br><br>
### GPU Prefetch Cap Reason
![Main screen](screenshots/Prefetch.png)


## Future Release Plans
* Add a better time filtering, perhaps with range
* Add dates range above the charts (since Streamlit plots have difficulty with dates)
* Add more analytics

## Project
This project is developed and maintained by [Shay Amram](https://www.linkedin.com/in/shay-amram-69924051/)
