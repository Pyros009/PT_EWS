# Data analytics in earthquake analysis
##     using historic data to create an early-warning model

### Project overview:
Portugal has been shaken by strong earthquakes() through time. Although earthquakes cannot be predicted, there's been many cases of successful early-warning system implementations across the globe, which contribute greatly to the mitigation of effects of these types of events.

Using data analysis and machine-learning tools, I attempt to create a model that predicts if there's a strong wave approaching and triggers a warning and try to implement it in (near) real-time using streaming systems like IRISDMC.

By using real historical data from IPMA (www.ipma.pt), USGS and earthquake websites coupled with data processing and integration, an event database was created to use as foundations to build data on.

Real wavedata was acquired by prompting IRISDMC servers on info from seismic stations belonging to Instituto D. Luis (IDL-FCUL) using Obspy module. A script was created to automatically clean (as much as possible) the wave data and retrieve important parameters.

All the information acquired was then passed into a machine-learning algorithm to try and predict if an incomming wave would be strong enough to be sensed (intensity scale > III).

### Major obstacles:

There were many challenges on this project. From the data gathering (mostly obtained from public .pdf catalogs from IPMA) to figuring out how to retrieve all the data from real-time and past events using Obspy, there were many hills to climb.

Wave data cleanup required multiple attempts just to scratch the bare minimum to be serviceable and decisions on how to acquire metrics of the wave also posed a challenge.

The biggest wall was how to implement (near) real-time processing. Python follows a linear (top-to-bottom) structure, which means that while it's gathering data, it cannot process it and vice-versa. Since data didn't come within a specific time-interval nor was predictable on arrival time, multi-processing had to be implemented (with data transfering between them).

### Conclusions:






