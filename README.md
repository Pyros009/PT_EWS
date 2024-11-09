# Data analytics in earthquake analysis
## using historic data to create an early-warning model

This is my first steps into building an earthquake early warning system by using Data Analysis and machine-learning to attempt to create a model that predicts if there's a strong wave approaching.

By using real historical data from IPMA (www.ipma.pt), USGS and earthquake websites coupled with data processing and integration, an event database was created to use as foundations to build data on.

Real wavedata was acquired by prompting IRISDMC servers on info from seismic stations belonging to Instituto D. Luis (IDL-FCUL) using Obspy module. A script was created to automatically clean (as much as possible) the wave data and retrieve important parameters.

All the information acquired was then passed into a machine-learning algorithm to try and predict if an incomming wave would be strong enough to be sensed (intensity scale > III).



