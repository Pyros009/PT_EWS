# Data analytics in earthquake analysis
##     using historic data to create an early-warning model

### Project overview:
Portugal has been shaken by strong [earthquakes](https://en.wikipedia.org/wiki/List_of_earthquakes_in_Portugal)([1755](https://en.wikipedia.org/wiki/1755_Lisbon_earthquake), [1909](https://en.wikipedia.org/wiki/1909_Benavente_earthquake), [1969](https://en.wikipedia.org/wiki/1969_Portugal_earthquake), etc) through time. Although earthquakes cannot be predicted, there's been many cases of successful early-warning system implementations across the globe, which contribute greatly to the mitigation of effects of these types of events.

Using data analysis and machine-learning tools, I attempt to create a model that predicts if there's a strong wave approaching and triggers a warning and try to implement it in (near) real-time using streaming systems like IRISDMC.

By using real historical data from [IPMA](https://www.ipma.pt/pt/geofisica/sismicidade/), [USGS]() and [Earthquakes]() websites coupled with data processing and integration, an event database was created to use as foundations to build data on.

Real wavedata was acquired by prompting IRISDMC servers on info from seismic stations belonging to Instituto D. Luis (IDL-FCUL) using Obspy module. A script was created to automatically clean (as much as possible) the wave data and retrieve important parameters.

All the information acquired was then passed into a machine-learning algorithm to try and predict if an incomming wave would be strong enough to be sensed (intensity scale > III).

### Major obstacles:

There were many challenges on this project. From the data gathering (mostly obtained from public .pdf catalogs from IPMA) to figuring out how to retrieve all the data from real-time and past events using Obspy, there were many hills to climb.

Wave data cleanup required multiple attempts just to scratch the bare minimum to be serviceable and decisions on how to acquire metrics of the wave also posed a challenge.

The biggest wall was how to implement (near) real-time processing. Python follows a linear (top-to-bottom) structure, which means that while it's gathering data, it cannot process it and vice-versa. Since data didn't come within a specific time-interval nor was predictable on arrival time, multi-processing had to be implemented (with data transfering between them).

### Conclusions:



### Next steps:


### Deliverables:

Presentation - [Gdocs presentation](https://docs.google.com/presentation/d/1mfNBHb25qzymuxEZuQ3XqDP8nWq9iWIRDuMScPlRra4/edit?usp=sharing) <br>
Tableau - [Overview of 2009-2023 Earthquakes in tableau](https://public.tableau.com/views/Overview_17309892235290/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)<br>
Kanban - [Work management](https://trello.com/invite/b/6717f6d1cbc0a1676b4a993f/ATTI641475a5e753919737745911417fa1b1302B2F1C/ews)<br>


