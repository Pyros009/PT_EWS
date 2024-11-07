from obspy.clients.seedlink.easyseedlink import create_client
from obspy.clients.fdsn import Client as FDSNclient
from obspy.core.inventory import Inventory
import time
from obspy import UTCDateTime
from bs4 import BeautifulSoup
import requests
import logging.handlers
import numpy as np
import logging
from multiprocessing import Queue

 # Configure logging (if not already configured in your main module)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataFetcher(Inventory):
    
    def __init__(self, *args, **kwargs):
        # Initialize both parent classes
        super().__init__()     
           
    def fetch_data(self,queue, url, source):
        logging.info(f'Fetching data from {url}') 
    
        try:   
            client = FDSNclient()
            
            def handle_data(trace):
                logging.info('P1-Received trace, generating response...')
                seedID = trace.get_id().split('.')
                network = seedID[0]
                station = seedID[1]
                location = "--"
                channel = seedID[3]
                
                # Define start and end times
                starttime = UTCDateTime.now() - 120  # 2 minute ago
                endtime = UTCDateTime.now()+1  # forever
                         
                # Fetch the inventory
                trace = client.get_waveforms(network=network,station=station,location=location,channel=channel, starttime=starttime, endtime=endtime, attach_response=True)
          
                logging.info(f'Sending {trace} to P3.')
                 
                # Send trace through the pipe to P3
                queue.put(trace)
            
            # Connect to the SeedLink server
            seedlink_client = create_client(f"{url}", handle_data)
            logging.info('Attempting to connect to SeedLink server')

            # Attempt to print available streams
            for server, values in source.items():
                for channel, ch in values.items():
                    for i in ch:
                        logging.info(f'Selecting stream: {server} - {channel} - {i}')
                        seedlink_client.select_stream(server, channel, i)  # Modify this based on actual available streams
            
            logging.info('Running client to receive data')
            seedlink_client.run()
                
        except Exception as e:
            logging.error(f"Error in fetch_data: {e}", exc_info=True)  # Log with traceback
