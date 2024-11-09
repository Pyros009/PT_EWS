# Data analytics in earthquake analysis
##     using historic data to create an early-warning model

### Project overview:
Portugal has been shaken by strong [earthquakes](https://en.wikipedia.org/wiki/List_of_earthquakes_in_Portugal) ([1755](https://en.wikipedia.org/wiki/1755_Lisbon_earthquake), [1909](https://en.wikipedia.org/wiki/1909_Benavente_earthquake), [1969](https://en.wikipedia.org/wiki/1969_Portugal_earthquake), etc) through time. Although earthquakes cannot be predicted, there's been many cases of successful early-warning system implementations across the globe, which contribute greatly to the mitigation of effects of these types of events.

Using data analysis and machine-learning tools, this project attempts to create a model that predicts if there's a strong wave approaching and triggers a warning and try to implement it in (near) real-time using streaming systems like IRISDMC.

By using real historical data from [IPMA](https://www.ipma.pt/pt/geofisica/sismicidade/), [USGS](https://earthquake.usgs.gov/earthquakes/map/?extent=20.67391,-37.00195&extent=51.17934,33.31055&range=search&sort=oldest&timeZone=utc&search=%7B%22name%22:%22Search%20Results%22,%22params%22:%7B%22starttime%22:%222005-01-01%2000:00:00%22,%22endtime%22:%222024-10-29%2023:59:59%22,%22maxlatitude%22:46.468,%22minlatitude%22:30.487,%22maxlongitude%22:2.197,%22minlongitude%22:-23.687,%22minmagnitude%22:-2,%22orderby%22:%22time%22%7D%7D) and [EarthquakeList](https://earthquakelist.org/portugal/) websites coupled with data processing and integration, an event database was created to use as foundations to build data on.

Real wavedata was acquired by prompting [IRISDMC/Earthscope](https://www.earthscope.org/) servers on info from seismic stations belonging to Instituto D. Luis ([IDL-FCUL](https://idl.ciencias.ulisboa.pt/)) using Obspy module. A script was created to automatically clean (as much as possible) the wave data and retrieve important parameters.

All the information acquired was then passed into a machine-learning algorithm to try and predict if an incomming wave would be strong enough to be sensed (intensity scale > III).

### Major obstacles:

There were many challenges on this project. From the data gathering (mostly obtained from public .pdf catalogs from IPMA) to figuring out how to retrieve all the data from real-time and past events using Obspy, there were many hills to climb.

Wave data cleanup required multiple attempts just to scratch the bare minimum to be serviceable and decisions on how to acquire metrics of the wave also posed a challenge.

The biggest wall was how to implement (near) real-time processing. Python follows a linear (top-to-bottom) structure, which means that while it's gathering data, it cannot process it and vice-versa. Since data didn't come within a specific time-interval nor was predictable on arrival time, multi-processing had to be implemented (with data transfering between them).

### Conclusions:
Although Machine-learning the project allowed the creation of a model (acc 78%, precision 60%) to predict if an incomming event was successful, the accuracy and recall values of the model are still too unreliable for deployment. Real-time data acquisition and processing can be done, but refinements on speed and capacity of the program still need to be aprimorated.

As conclusion, the project indeed points towards the viability of developing a machine-learning algorithm to predict incomming strong-motion waves (large enough to be sensed).

### Next steps:
Further refinement of the model is a must to achieve reliability. This needs to be preceded by further fine-tuning of historical data values, in order to create a much more robust historic database from which our model learns. Refining some hyperparameters might also prove advantageous once the database is aprimorated.

Strengthening the real-time data acquisition from multiple stations (and the relevant processing of data) needs to be increased. This will provide much fiability to data acquired but also allow for a wider usage of the model.

Multi-threading the processing might help fix some bottlenecking on processing data which will be critical once retrieving data from multiple stations at the same time.

### Deliverables:

Presentation - [Gdocs presentation](https://docs.google.com/presentation/d/1mfNBHb25qzymuxEZuQ3XqDP8nWq9iWIRDuMScPlRra4/edit?usp=sharing) <br>
Tableau - [Overview of 2009-2023 Earthquakes in tableau](https://public.tableau.com/views/Overview_17309892235290/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)<br>
Kanban - [Work management](https://trello.com/invite/b/6717f6d1cbc0a1676b4a993f/ATTI641475a5e753919737745911417fa1b1302B2F1C/ews)<br>

### Some Publications: 

- [Somoza L, Medialdea T, Terrinha P, Ramos A and Vázquez J-T (2021) - Submarine Active Faults and MorphoTectonics Around the Iberian Margins: Seismic and Tsunamis Hazards. Front. Earth Sci. 9:653639. doi: 10.3389/feart.2021.653639](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2021.653639/full)<br>
- [Havskov J. & Ottemöller L. (2009) - Processing Earthquake Data ](https://www.geo.uib.no/seismo/SOFTWARE/DOCUMENTATION/processing_earthquake_data.pdf)<br>
- [Bhardwaj R., Kumar A., Sharma M. L. (2012) - Analysis of Tauc (τc) and Pd attributes for Earthquake Early Warning in India](https://www.iitk.ac.in/nicee/wcee/article/WCEE2012_0696.pdf)<br>
- [Tsuboi S., Abe K., Takano K., Yamanaka Y. (1995) - Rapid Determination of Mw from Broadband P Waveforms](https://pubs.geoscienceworld.org/ssa/bssa/article/85/2/606/119958/Rapid-determination-of-Mw-from-broadband-P)<br>
- [Wu Y.M., Hsiao N., Lee W., Teng T. (2007) - State of the Art and Progress in the Earthquake Early Warning System in Taiwan](https://www.researchgate.net/publication/238676886_14_State_of_the_Art_and_Progress_in_the_Earthquake_Early_Warning_System_in_Taiwan)<br>